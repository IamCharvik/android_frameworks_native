/*
 * Copyright 2025 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS

#include "KawaseBlurDualFilterV2.h"
#include <SkAlphaType.h>
#include <SkBlendMode.h>
#include <SkCanvas.h>
#include <SkData.h>
#include <SkPaint.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkShader.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <android-base/properties.h>
#include <include/gpu/GpuTypes.h>
#include <include/gpu/ganesh/SkSurfaceGanesh.h>
#include <log/log.h>
#include <utils/Trace.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {
namespace {

constexpr const char* kV2InputScaleProperty = "persist.sys.sf.gb_scale";

float readV2InputScale() {
    const std::string value = base::GetProperty(kV2InputScaleProperty, "");
    if (!value.empty()) {
        char* end = nullptr;
        const float scale = std::strtof(value.c_str(), &end);
        if (end != value.c_str() && std::isfinite(scale)) {
            return std::clamp(scale, 0.05f, 0.25f);
        }
    }
    return 0.1667f;
}

} // namespace

// Samples each vertex of a diamond using a total of 4 samples.
// This shader is used for the initial 4x down-sampling pass, with the sampling offset
// handpicked to minimize aliasing.

const SkString kEffectSource_KawaseBlurDualFilterV2_QuarterResDownSampleBlurEffect(R"(
    uniform shader child;

    const float2 STEP_0 = float2( 0.25, 0.25);
    const float2 STEP_1 = float2( 0.25, -0.25);
    const float2 STEP_2 = float2(-0.25, -0.25);
    const float2 STEP_3 = float2(-0.25, 0.25);

    half4 main(float2 xy) {
        half3 c = child.eval(xy + STEP_0).rgb;
        c += child.eval(xy + STEP_1).rgb;
        c += child.eval(xy + STEP_2).rgb;
        c += child.eval(xy + STEP_3).rgb;

        return half4(c * 0.25, 1.0);
    }
)");

// Samples each vertex of a diamond plus the center pixel, using a total of 5 samples.
// This shader is used for the 2x down-sampling passes after the initial pass.
const SkString kEffectSource_KawaseBlurDualFilterV2_HalfResDownSampleBlurEffect(R"(
    uniform shader child;

    const float2 STEP_0 = float2( 0.5, 0.5);
    const float2 STEP_1 = float2( 0.5, -0.5);
    const float2 STEP_2 = float2(-0.5, -0.5);
    const float2 STEP_3 = float2(-0.5, 0.5);

    half4 main(float2 xy) {
        half3 c = child.eval(xy).rgb * 4.0;
        c += child.eval(xy + STEP_0).rgb;
        c += child.eval(xy + STEP_1).rgb;
        c += child.eval(xy + STEP_2).rgb;
        c += child.eval(xy + STEP_3).rgb;

        return half4(c * 0.125, 1.0);
    }
)");

// A shader to sample each vertex of a unit regular heptagon, plus the original fragment
// coordinate, using a total of 8 samples.
const SkString kEffectSource_KawaseBlurDualFilterV2_UpSampleBlurEffect(R"(
    uniform shader child;
    uniform float in_blurOffset;
    uniform float in_crossFade;
    uniform float in_weightedCrossFade;

    const float2 STEP_0 = float2( 1.0, 0.0);
    const float2 STEP_1 = float2( 0.623489802,  0.781831482);
    const float2 STEP_2 = float2(-0.222520934,  0.974927912);
    const float2 STEP_3 = float2(-0.900968868,  0.433883739);
    const float2 STEP_4 = float2( 0.900968868, -0.433883739);
    const float2 STEP_5 = float2(-0.222520934, -0.974927912);
    const float2 STEP_6 = float2(-0.623489802, -0.781831482);

    half4 main(float2 xy) {
        half3 c = child.eval(xy).rgb;

        c += child.eval(xy + STEP_0 * in_blurOffset).rgb;
        c += child.eval(xy + STEP_1 * in_blurOffset).rgb;
        c += child.eval(xy + STEP_2 * in_blurOffset).rgb;
        c += child.eval(xy + STEP_3 * in_blurOffset).rgb;
        c += child.eval(xy + STEP_4 * in_blurOffset).rgb;
        c += child.eval(xy + STEP_5 * in_blurOffset).rgb;
        c += child.eval(xy + STEP_6 * in_blurOffset).rgb;

        return half4(c * in_weightedCrossFade, in_crossFade);
    }
)");

KawaseBlurDualFilterV2::KawaseBlurDualFilterV2(RuntimeEffectManager& effectManager)
      : BlurFilter(effectManager), mInputScale(readV2InputScale()) {
    mQuarterResDownSampleBlurEffect =
            effectManager.mKnownEffects[kKawaseBlurDualFilterV2_QuarterResDownSampleBlurEffect];
    mHalfResDownSampleBlurEffect =
            effectManager.mKnownEffects[kKawaseBlurDualFilterV2_HalfResDownSampleBlurEffect];
    mUpSampleBlurEffect = effectManager.mKnownEffects[kKawaseBlurDualFilterV2_UpSampleBlurEffect];
}

sk_sp<SkSurface> KawaseBlurDualFilterV2::obtainSurface(SkiaGpuContext* context,
                                                       const SkImageInfo& info,
                                                       int index) const {
    if (index < 0 || index >= kMaxSurfaces || !context) {
        return nullptr;
    }

    auto& pool = mPools[index];
    size_t& count = mCounts[index];

    const uint64_t minAvailableFrame = mFrameCounter >= 2 ? mFrameCounter - 2 : 0;
    SurfaceSlot* candidate = nullptr;
    int staleSlotIndex = -1;
    for (size_t i = 0; i < count; ++i) {
        auto& slot = pool[i];
        if (!slot.surface || slot.context != context) continue;
        if (!(slot.info == info)) {
            if (staleSlotIndex < 0 && slot.lastUsedFrame < minAvailableFrame) {
                staleSlotIndex = static_cast<int>(i);
            }
            continue;
        }
        if (slot.lastUsedFrame < minAvailableFrame) {
            slot.lastUsedFrame = mFrameCounter;
            return slot.surface;
        }
        if (!candidate) candidate = &slot;
    }
    if (candidate && count >= kPoolCapacity) {
        candidate->lastUsedFrame = mFrameCounter;
        return candidate->surface;
    }

    ATRACE_NAME("KawaseV2SurfaceCreate");
    sk_sp<SkSurface> surface = context->createRenderTarget(info);
    if (!surface) {
        return nullptr;
    }

    if (staleSlotIndex >= 0) {
        pool[staleSlotIndex] = {info, context, surface, mFrameCounter};
    } else if (count < kPoolCapacity) {
        pool[count] = {info, context, surface, mFrameCounter};
        ++count;
    } else {
        size_t lruIndex = 0;
        uint64_t oldest = pool[0].lastUsedFrame;
        for (size_t i = 1; i < count; ++i) {
            if (pool[i].lastUsedFrame < oldest) {
                oldest = pool[i].lastUsedFrame;
                lruIndex = i;
            }
        }
        pool[lruIndex] = {info, context, surface, mFrameCounter};
    }
    return surface;
}

void KawaseBlurDualFilterV2::blurInto(const sk_sp<SkSurface>& drawSurface,
                                      const sk_sp<SkImage>& readImage, const float radius,
                                      const float alpha,
                                      const sk_sp<SkRuntimeEffect>& blurEffect) const {
    const float scale = static_cast<float>(drawSurface->width()) / readImage->width();
    SkMatrix blurMatrix = SkMatrix::Scale(scale, scale);
    blurInto(drawSurface,
             readImage->makeShader(SkTileMode::kClamp, SkTileMode::kClamp,
                                   SkSamplingOptions(SkFilterMode::kLinear, SkMipmapMode::kNone),
                                   blurMatrix),
             radius, alpha, blurEffect);
}

void KawaseBlurDualFilterV2::blurInto(const sk_sp<SkSurface>& drawSurface, sk_sp<SkShader> input,
                                      const float radius, const float alpha,
                                      const sk_sp<SkRuntimeEffect>& blurEffect) const {
    SkPaint paint;
    if (blurEffect == mUpSampleBlurEffect) {
        if (radius == 0) {
            paint.setShader(std::move(input));
            paint.setAlphaf(alpha);
        } else {
            SkRuntimeShaderBuilder blurBuilder(blurEffect);
            blurBuilder.child("child") = std::move(input);
            if (blurEffect == mUpSampleBlurEffect) {
                blurBuilder.uniform("in_crossFade") = alpha;
                blurBuilder.uniform("in_weightedCrossFade") = alpha * 0.125f;
                blurBuilder.uniform("in_blurOffset") = radius;
            }
            paint.setShader(blurBuilder.makeShader(nullptr));
        }
    } else {
        SkRuntimeShaderBuilder blurBuilder(blurEffect);
        blurBuilder.child("child") = std::move(input);
        paint.setShader(blurBuilder.makeShader(nullptr));
    }
    paint.setBlendMode(alpha == 1.0f ? SkBlendMode::kSrc : SkBlendMode::kSrcOver);
    if (alpha == 1.0f) {
        drawSurface->getCanvas()->discard();
    }
    drawSurface->getCanvas()->drawPaint(paint);
}

sk_sp<SkImage> KawaseBlurDualFilterV2::generate(SkiaGpuContext* context, const uint32_t blurRadius,
                                                const sk_sp<SkImage> input,
                                                const SkRect& blurRect) const {
    if (!context || !input || blurRadius == 0 || blurRect.isEmpty()) {
        return input;
    }
    ++mFrameCounter;

    const float inputScale = mInputScale;

    // Apply a conversion factor of (1 / sqrt(3)) to match Skia's built-in blur as used by
    // RenderEffect. See the comment in SkBlurMask.cpp for reasoning behind this.
    const float radius = blurRadius * 0.57735f;

    SkIRect targetBlurRect;
    blurRect.roundOut(&targetBlurRect);
    const int rawW0 = std::max(
            1, static_cast<int>(std::round((targetBlurRect.width() + 4) * inputScale)));
    const int rawH0 = std::max(
            1, static_cast<int>(std::round((targetBlurRect.height() + 4) * inputScale)));
    const int w0 = std::max(32, (rawW0 + 31) & ~31);
    const int h0 = std::max(32, (rawH0 + 31) & ~31);

    int maxPasses = kMaxSurfaces - 1;
    while (maxPasses > 1 && ((w0 >> maxPasses) < 16 || (h0 >> maxPasses) < 16)) {
        --maxPasses;
    }

    // Use a variable number of blur passes depending on the radius. The non-integer part of this
    // calculation is used to mix the final pass into the second-last with an alpha blend.
    const float filterDepth =
            std::min(static_cast<float>(maxPasses), radius * inputScale / 2.5f);
    const int filterPasses = std::min(maxPasses, static_cast<int>(ceil(filterDepth)));

    auto makeSurface = [&](int index) -> sk_sp<SkSurface> {
        const int newW = std::max(1, w0 >> index);
        const int newH = std::max(1, h0 >> index);
        SkImageInfo info = input->imageInfo().makeWH(newW, newH).makeAlphaType(kOpaque_SkAlphaType);
        if (info.colorType() == kRGBA_F16_SkColorType) {
            info = info.makeColorType(kRGBA_8888_SkColorType);
        }
        return obtainSurface(context, info, index);
    };

    // Render into surfaces downscaled by 1x, 2x, 4x and 8x from the initial downscale.
    sk_sp<SkSurface> surfaces[kMaxSurfaces] = {nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i <= filterPasses; i++) {
        surfaces[i] = makeSurface(i);
        if (!surfaces[i]) {
            return input;
        }
    }

    // Kawase is an approximation of Gaussian, but behaves differently because it is made up of many
    // simpler blurs. A transformation is required to approximate the same effect as Gaussian.
    float sumSquaredR = 0;
    float sumSquaredStep = 0;
    for (int i = 0; i < filterPasses; i++) {
        const float alpha = std::min(1.0f, filterDepth - i);
        sumSquaredR += powf(powf(2.0f, i - 1) * alpha * M_SQRT2, 2.0f);
        sumSquaredStep += powf(powf(2.0f, i) * alpha, 2.0f);
    }
    // Solve for R = sqrt(sum(r_i^2)).
    float step = sqrt(max(0.0f, powf(radius * inputScale, 2) - sumSquaredR) /
                      (sumSquaredStep == 0 ? 1 : sumSquaredStep));

    // Start by downscaling and doing the first blur pass
    {
        // For sampling Skia's API expects the inverse of what logically seems appropriate. In this
        // case one may expect Translate(blurRect.fLeft, blurRect.fTop) * Scale(1/inputScale)
        // but instead we must do the inverse.
        SkMatrix blurMatrix = SkMatrix::Translate(-blurRect.fLeft, -blurRect.fTop);
        blurMatrix.postScale(inputScale, inputScale);
        const auto sourceShader =
                input->makeShader(SkTileMode::kClamp, SkTileMode::kClamp,
                                  SkSamplingOptions(SkFilterMode::kLinear, SkMipmapMode::kNone),
                                  blurMatrix);
        blurInto(surfaces[0], std::move(sourceShader),
                 0, // unused blur radius. The blur effect hardcodes the radius.
                 1.0f, mQuarterResDownSampleBlurEffect);
    }

    // Next the remaining downscale blur passes.
    for (int i = 0; i < filterPasses; i++) {
        sk_sp<SkImage> tmp = surfaces[i]->makeTemporaryImage();
        if (!tmp) {
            return input;
        }
        blurInto(surfaces[i + 1], tmp,
                 0, // unused blur radius. The blur effect hardcodes the radius.
                 1.0f, mHalfResDownSampleBlurEffect);
    }
    // Finally blur+upscale back to our original size.
    for (int i = filterPasses - 1; i >= 0; i--) {
        sk_sp<SkImage> tmp = surfaces[i + 1]->makeTemporaryImage();
        if (!tmp) {
            return input;
        }
        blurInto(surfaces[i], tmp, step, std::min(1.0f, filterDepth - i), mUpSampleBlurEffect);
    }

    sk_sp<SkImage> result = surfaces[0]->makeImageSnapshot();
    return result ? result : input;
}

} // namespace skia
} // namespace renderengine
} // namespace android
