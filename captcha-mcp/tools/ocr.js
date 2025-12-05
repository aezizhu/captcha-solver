/**
 * OCR Tools - Enhanced Local Text Extraction
 * High-accuracy OCR with advanced preprocessing and multi-pass recognition
 */

import Tesseract from 'tesseract.js';
import sharp from 'sharp';

/**
 * Preprocess image for better OCR accuracy
 * @param {string} imageBase64 - Base64 encoded image
 * @param {object} options - Preprocessing options
 * @returns {Promise<string>} Preprocessed base64 image
 */
async function preprocessForOCR(imageBase64, options = {}) {
    const {
        grayscale = true,
        contrast = 1.5,
        sharpen = true,
        threshold = null,
        invert = false,
        scale = 2,
        denoise = true
    } = options;

    try {
        const buffer = Buffer.from(imageBase64, 'base64');
        let pipeline = sharp(buffer);

        // Get metadata for smart processing
        const metadata = await sharp(buffer).metadata();

        // Scale up small images (helps OCR)
        if (metadata.width < 200 || metadata.height < 50) {
            pipeline = pipeline.resize(
                Math.max(metadata.width * scale, 200),
                null,
                { kernel: 'lanczos3' }
            );
        }

        // Convert to grayscale
        if (grayscale) {
            pipeline = pipeline.grayscale();
        }

        // Increase contrast
        if (contrast !== 1) {
            pipeline = pipeline.linear(contrast, -(128 * (contrast - 1)));
        }

        // Sharpen to make text edges clearer
        if (sharpen) {
            pipeline = pipeline.sharpen({ sigma: 1.5 });
        }

        // Denoise
        if (denoise) {
            pipeline = pipeline.median(1);
        }

        // Apply threshold for binary image (good for captchas)
        if (threshold !== null) {
            pipeline = pipeline.threshold(threshold);
        }

        // Invert if needed (white text on dark background)
        if (invert) {
            pipeline = pipeline.negate();
        }

        const outputBuffer = await pipeline.png().toBuffer();
        return outputBuffer.toString('base64');
    } catch (error) {
        // Return original if preprocessing fails
        return imageBase64;
    }
}

/**
 * Perform OCR on a base64 encoded image
 * @param {string} imageBase64 - Base64 encoded image (without data: prefix)
 * @param {string} lang - Language code (e.g., 'eng', 'chi_sim')
 * @returns {Promise<object>} OCR result with text and confidence
 */
export async function performOCR(imageBase64, lang = 'eng') {
    try {
        const dataUrl = `data:image/png;base64,${imageBase64}`;

        const result = await Tesseract.recognize(
            dataUrl,
            lang,
            { logger: () => { } }
        );

        return {
            success: true,
            text: result.data.text.trim(),
            confidence: result.data.confidence,
            words: result.data.words.map(w => ({
                text: w.text,
                confidence: w.confidence,
                bbox: w.bbox
            }))
        };
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Correct common OCR mistakes in captcha text
 * Conservative mode: only fix obvious issues
 * Aggressive mode: for numeric-only captchas
 */
function correctOCRErrors(text, aggressive = false) {
    let result = text
        .trim()
        .replace(/\s+/g, '')           // Remove whitespace
        .replace(/['"`,.\-_]/g, '')    // Remove punctuation
        .replace(/\n/g, '');           // Remove newlines

    // Always fix these obvious confusions
    result = result
        .replace(/[|]/g, '1')          // | -> 1
        .replace(/[lI]/g, (match, offset, str) => {
            // Only convert l/I to 1 if surrounded by numbers
            const prev = str[offset - 1];
            const next = str[offset + 1];
            if (/\d/.test(prev) || /\d/.test(next)) return '1';
            return match;
        });

    // Aggressive mode for numeric captchas only
    if (aggressive) {
        result = result
            .replace(/[oO]/g, '0')
            .replace(/[sS]/g, '5')
            .replace(/[zZ]/g, '2')
            .replace(/[bB]/g, '8')
            .replace(/[gG]/g, '9');
    }

    return result;
}

/**
 * Enhanced captcha OCR with multiple preprocessing attempts
 * @param {string} imageBase64 - Base64 encoded image
 * @param {object} options - Processing options
 * @returns {Promise<object>} Best OCR result
 */
export async function performCaptchaOCR(imageBase64, options = {}) {
    const {
        lang = 'eng',
        whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        psm = 7,                    // Page segmentation mode (7 = single line)
        multiPass = true            // Try multiple preprocessing settings
    } = options;

    // Different preprocessing configurations to try
    const preprocessConfigs = multiPass ? [
        { name: 'default', grayscale: true, contrast: 1.3, sharpen: true, threshold: null },
        { name: 'high-contrast', grayscale: true, contrast: 2.0, sharpen: true, threshold: 128 },
        { name: 'binary', grayscale: true, contrast: 1.5, sharpen: true, threshold: 100 },
        { name: 'inverted', grayscale: true, contrast: 1.5, sharpen: true, threshold: 128, invert: true },
        { name: 'soft', grayscale: true, contrast: 1.2, sharpen: false, threshold: null }
    ] : [
        { name: 'default', grayscale: true, contrast: 1.3, sharpen: true, threshold: null }
    ];

    let bestResult = null;
    let bestConfidence = 0;
    const allAttempts = [];

    for (const config of preprocessConfigs) {
        try {
            // Preprocess image
            const processedImage = await preprocessForOCR(imageBase64, config);
            const dataUrl = `data:image/png;base64,${processedImage}`;

            // Create worker and configure
            const worker = await Tesseract.createWorker(lang);
            await worker.setParameters({
                tessedit_char_whitelist: whitelist,
                tessedit_pageseg_mode: psm.toString()
            });

            // Recognize
            const result = await worker.recognize(dataUrl);
            await worker.terminate();

            const rawText = result.data.text.trim();
            const confidence = result.data.confidence;
            const cleanedText = correctOCRErrors(rawText);

            allAttempts.push({
                config: config.name,
                text: cleanedText,
                rawText: rawText,
                confidence: confidence
            });

            // Track best result
            if (confidence > bestConfidence && cleanedText.length > 0) {
                bestConfidence = confidence;
                bestResult = {
                    success: true,
                    text: cleanedText,
                    rawText: rawText,
                    confidence: confidence,
                    usedConfig: config.name
                };
            }
        } catch (error) {
            allAttempts.push({
                config: config.name,
                error: error.message
            });
        }
    }

    if (bestResult) {
        bestResult.allAttempts = allAttempts;
        return bestResult;
    }

    return {
        success: false,
        error: 'All OCR attempts failed',
        allAttempts: allAttempts
    };
}

/**
 * Extract math expression from image and evaluate
 * Enhanced with better preprocessing and expression parsing
 * @param {string} imageBase64 - Base64 encoded image
 * @returns {Promise<object>} Math result
 */
export async function solveMathCaptchaLocally(imageBase64) {
    try {
        // Use specialized math whitelist
        const ocrResult = await performCaptchaOCR(imageBase64, {
            whitelist: '0123456789+-*×÷=/xX?= ',
            psm: 7,
            multiPass: true
        });

        if (!ocrResult.success) {
            return ocrResult;
        }

        // Parse and normalize the expression
        let expression = ocrResult.text
            .replace(/[×xX]/g, '*')     // Multiplication
            .replace(/÷/g, '/')         // Division
            .replace(/[=?]/g, '')       // Remove = and ?
            .replace(/\s/g, '')         // Remove spaces
            .trim();

        // Try to extract just the math part
        const mathMatch = expression.match(/(\d+)\s*([+\-*/])\s*(\d+)/);
        if (mathMatch) {
            expression = `${mathMatch[1]}${mathMatch[2]}${mathMatch[3]}`;
        }

        // Validate expression
        if (!/^[\d+\-*/().]+$/.test(expression)) {
            return {
                success: false,
                error: 'Could not parse math expression',
                detected: ocrResult.text,
                cleanedExpression: expression
            };
        }

        // Evaluate
        try {
            const result = Function(`"use strict"; return (${expression})`)();
            const roundedResult = Math.round(result);

            return {
                success: true,
                expression: expression,
                result: roundedResult.toString(),
                ocrConfidence: ocrResult.confidence,
                ocrConfig: ocrResult.usedConfig
            };
        } catch (evalError) {
            return {
                success: false,
                error: 'Could not evaluate expression',
                detected: ocrResult.text,
                expression: expression
            };
        }
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Attempt OCR with automatic retry and fallback
 * Best effort for maximum accuracy
 */
export async function bestEffortOCR(imageBase64, options = {}) {
    const { expectedLength, numericOnly = false, alphaOnly = false } = options;

    // First try with standard settings
    let result = await performCaptchaOCR(imageBase64, {
        whitelist: numericOnly ? '0123456789' :
            alphaOnly ? 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' :
                'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        multiPass: true
    });

    // Validate result length if expected
    if (result.success && expectedLength && result.text.length !== expectedLength) {
        // Try again with different PSM modes
        for (const psm of [8, 6, 13]) {
            const retry = await performCaptchaOCR(imageBase64, {
                whitelist: result.text.length > 0 ? undefined : 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                psm: psm,
                multiPass: true
            });

            if (retry.success && retry.text.length === expectedLength) {
                return retry;
            }

            if (retry.success && retry.confidence > result.confidence) {
                result = retry;
            }
        }
    }

    return result;
}
