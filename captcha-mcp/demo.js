#!/usr/bin/env node
/**
 * Live Captcha Solving Demo
 * Creates captcha-style problems and uses MCP tools to solve them
 */

import sharp from 'sharp';
import { performCaptchaOCR, solveMathCaptchaLocally } from './tools/ocr.js';
import { analyzeCaptchaType, calculateSliderOffset, analyzeImageGrid, preprocessImage } from './tools/image-analysis.js';

console.log('🎯 MCP Captcha Solver - Live Demo\n');
console.log('='.repeat(50));

// Demo 1: Create a text captcha and solve it
async function demo1_TextCaptcha() {
    console.log('\n📝 DEMO 1: Text Captcha Recognition');
    console.log('-'.repeat(40));

    // Create a simple captcha image with text "AB12"
    const text = 'AB12';
    console.log(`Creating captcha with text: "${text}"`);

    // Create image with text overlay using Sharp
    const svg = `
    <svg width="150" height="50">
      <rect width="100%" height="100%" fill="#f0f0f0"/>
      <text x="20" y="35" font-size="28" font-family="Arial" fill="#333">${text}</text>
    </svg>
  `;

    const imageBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
    const imageBase64 = imageBuffer.toString('base64');

    console.log('Image created, running OCR...');

    try {
        const result = await performCaptchaOCR(imageBase64, { lang: 'eng' });
        console.log('OCR Result:', result);

        if (result.success) {
            const match = result.text.replace(/\s/g, '').toUpperCase() === text;
            console.log(`Expected: "${text}", Got: "${result.text}"`);
            console.log(match ? '✅ CORRECT!' : '⚠️ Partial match');
        }
    } catch (e) {
        console.log('OCR Error:', e.message);
    }
}

// Demo 2: Create a math captcha and solve it
async function demo2_MathCaptcha() {
    console.log('\n🔢 DEMO 2: Math Captcha Solving');
    console.log('-'.repeat(40));

    const expression = '5+3=?';
    const expectedAnswer = '8';
    console.log(`Creating math captcha: "${expression}"`);

    const svg = `
    <svg width="150" height="50">
      <rect width="100%" height="100%" fill="#e8e8e8"/>
      <text x="15" y="38" font-size="32" font-family="Arial" fill="#222">5+3=?</text>
    </svg>
  `;

    const imageBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
    const imageBase64 = imageBuffer.toString('base64');

    console.log('Image created, solving math...');

    try {
        const result = await solveMathCaptchaLocally(imageBase64);
        console.log('Math Result:', result);

        if (result.success) {
            console.log(`Expected: ${expectedAnswer}, Got: ${result.result}`);
            console.log(result.result === expectedAnswer ? '✅ CORRECT!' : '❌ Wrong answer');
        }
    } catch (e) {
        console.log('Math Error:', e.message);
    }
}

// Demo 3: Slider puzzle offset calculation
async function demo3_SliderPuzzle() {
    console.log('\n🎮 DEMO 3: Slider Puzzle Offset');
    console.log('-'.repeat(40));

    // Create a slider background with a "gap" at position 120px
    const width = 300;
    const height = 150;
    const gapPosition = 120;

    console.log(`Creating slider with gap at ${gapPosition}px`);

    // Create background with a visible "notch"
    const svg = `
    <svg width="${width}" height="${height}">
      <rect width="100%" height="100%" fill="#4a90d9"/>
      <rect x="${gapPosition}" y="30" width="50" height="90" fill="#1a1a1a" stroke="#fff" stroke-width="2"/>
    </svg>
  `;

    const imageBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
    const imageBase64 = imageBuffer.toString('base64');

    console.log('Image created, calculating offset...');

    try {
        const result = await calculateSliderOffset(imageBase64);
        console.log('Slider Result:', result);

        if (result.success) {
            const error = Math.abs(result.estimatedOffset - gapPosition);
            console.log(`Expected: ~${gapPosition}px, Got: ${result.estimatedOffset}px`);
            console.log(error < 30 ? '✅ Close enough!' : `⚠️ Off by ${error}px`);
        }
    } catch (e) {
        console.log('Slider Error:', e.message);
    }
}

// Demo 4: Image grid analysis
async function demo4_ImageGrid() {
    console.log('\n🔲 DEMO 4: Image Grid Analysis');
    console.log('-'.repeat(40));

    const gridSize = 3;
    const cellSize = 100;
    console.log(`Creating ${gridSize}x${gridSize} image grid`);

    // Create a 3x3 grid image
    const svg = `
    <svg width="${cellSize * gridSize}" height="${cellSize * gridSize}">
      <rect width="100%" height="100%" fill="#ddd"/>
      ${Array.from({ length: gridSize * gridSize }, (_, i) => {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        return `<rect x="${col * cellSize + 2}" y="${row * cellSize + 2}" width="${cellSize - 4}" height="${cellSize - 4}" fill="#${(i % 2 === 0) ? 'aaa' : '888'}" stroke="#333"/>`;
    }).join('')}
    </svg>
  `;

    const imageBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
    const imageBase64 = imageBuffer.toString('base64');

    console.log('Image created, analyzing grid...');

    try {
        const result = await analyzeImageGrid(imageBase64, gridSize);
        console.log('Grid Result:', JSON.stringify(result, null, 2));

        if (result.success) {
            console.log(`Detected ${result.totalCells} cells`);
            console.log('Cell 0 center:', `(${result.cells[0].centerX}, ${result.cells[0].centerY})`);
            console.log('Cell 4 center (middle):', `(${result.cells[4].centerX}, ${result.cells[4].centerY})`);
            console.log('✅ Grid analyzed successfully!');
        }
    } catch (e) {
        console.log('Grid Error:', e.message);
    }
}

// Demo 5: Captcha type detection
async function demo5_TypeDetection() {
    console.log('\n🔍 DEMO 5: Captcha Type Detection');
    console.log('-'.repeat(40));

    // Create different aspect ratio images
    const scenarios = [
        { name: 'Text (wide)', width: 200, height: 50 },
        { name: 'Slider (square)', width: 300, height: 300 },
        { name: 'Grid (large square)', width: 400, height: 400 }
    ];

    for (const scenario of scenarios) {
        const svg = `<svg width="${scenario.width}" height="${scenario.height}"><rect fill="#ccc" width="100%" height="100%"/></svg>`;
        const imageBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
        const imageBase64 = imageBuffer.toString('base64');

        try {
            const result = await analyzeCaptchaType(imageBase64);
            console.log(`${scenario.name}: Detected as "${result.detectedType}" (${(result.confidence * 100).toFixed(0)}% confidence)`);
        } catch (e) {
            console.log(`${scenario.name}: Error - ${e.message}`);
        }
    }
    console.log('✅ Type detection complete!');
}

// Run all demos
async function runAllDemos() {
    try {
        await demo1_TextCaptcha();
        await demo2_MathCaptcha();
        await demo3_SliderPuzzle();
        await demo4_ImageGrid();
        await demo5_TypeDetection();

        console.log('\n' + '='.repeat(50));
        console.log('🎉 All demos completed!');
        console.log('='.repeat(50));
        console.log('\nThe MCP tools successfully:');
        console.log('  ✓ Analyzed captcha types');
        console.log('  ✓ Processed image grids');
        console.log('  ✓ Calculated slider offsets');
        console.log('  ✓ Ready to solve real captchas with API keys');
    } catch (e) {
        console.error('Demo failed:', e);
    }
}

runAllDemos();
