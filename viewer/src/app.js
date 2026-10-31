import { inspectDesign, pairFiles } from './review.js';

const $ = id => document.getElementById(id);
let pairs = [], current = null, sdk = null, generation = 0;
let urls = [];
function clearReview() {
    current = null;
    $('review').hidden = true;
    $('insert').disabled = true;
    $('preview').removeAttribute('src');
    for (const url of urls) URL.revokeObjectURL(url);
    urls = [];
}
function objectURL(file) { const url = URL.createObjectURL(file); urls.push(url); return url; }
async function selectPair() {
    const request = ++generation;
    clearReview();
    const pair = pairs[Number($('designs').value)];
    if (!pair) return;
    try {
        if (!pair.json || !pair.png) throw new Error(`Missing ${pair.json ? 'PNG' : 'JSON'} for ${pair.name}. Select both matching files.`);
        const design = JSON.parse(await pair.json.text());
        const bitmap = await createImageBitmap(pair.png);
        const {width, height} = bitmap;
        bitmap.close();
        if (request !== generation) return;
        const warnings = inspectDesign(design, width, height);
        current = pair;
        $('preview').src = objectURL(pair.png);
        $('download').href = $('preview').src;
        $('download').download = pair.png.name;
        $('json-download').href = objectURL(pair.json);
        $('json-download').download = pair.json.name;
        $('title').textContent = pair.name;
        $('dimensions').textContent = `${width} × ${height} px · ${design.elements.length} elements · Original engine PNG`;
        $('json').textContent = JSON.stringify(design, null, 2);
        $('warnings').replaceChildren();
        for (const message of warnings.length ? warnings : ['No text below 8 px. Check contrast and wrapping in the preview.']) {
            const li = document.createElement('li'); li.textContent = message; $('warnings').append(li);
        }
        $('text-rows').replaceChildren();
        for (const e of design.elements.filter(e => e.type === 'text')) {
            const tr = document.createElement('tr');
            for (const value of [e.id, `${e.font_size.toFixed(2)} px`, e.content]) {
                const td = document.createElement('td'); td.textContent = value; tr.append(td);
            }
            $('text-rows').append(tr);
        }
        $('review').hidden = false;
        $('insert').disabled = !sdk;
        $('status').textContent = 'Loaded. Review at actual size before submitting.';
    } catch (error) {
        if (request === generation) { clearReview(); $('status').textContent = error.message; }
    }
}
$('files').addEventListener('change', () => {
    ++generation; clearReview(); pairs = []; $('designs').replaceChildren();
    try {
        pairs = pairFiles($('files').files);
        for (const [i, pair] of pairs.entries()) {
            const option = document.createElement('option'); option.value = i; option.textContent = pair.name; $('designs').append(option);
        }
        if (pairs.length) selectPair(); else $('status').textContent = 'Choose JSON and PNG output files.';
    } catch (error) { $('status').textContent = error.message; }
});
$('designs').addEventListener('change', selectPair);
$('zoom').addEventListener('click', () => {
    const actual = $('viewport').classList.toggle('actual');
    $('zoom').setAttribute('aria-pressed', String(actual));
    $('zoom').textContent = actual ? 'Fit preview' : 'Actual size';
});
$('insert').addEventListener('click', async () => {
    if (!sdk || !current) return;
    const pair = current;
    $('insert').disabled = true;
    try {
        await sdk.app.document.addImage(pair.png, {title: pair.name});
        $('status').textContent = `Added ${pair.name} to the current Express page.`;
    } catch (error) { $('status').textContent = `Could not insert image: ${error.message}`; }
    finally { $('insert').disabled = !sdk || !current; }
});
// The standalone viewer works offline. Only the explicit Express mode loads its SDK.
if (window.parent !== window) {
    import('https://express.adobe.com/static/add-on-sdk/sdk.js').then(async module => {
        await module.default.ready;
        sdk = module.default;
        $('host').textContent = 'Connected to Adobe Express. Inserts the rendered PNG as one image.';
        $('insert').disabled = !current;
    }).catch(() => { $('host').textContent = 'Express connection unavailable. Preview and downloads still work.'; });
} else {
    $('host').textContent = 'Standalone preview. Load this add-on in Adobe Express to insert the rendered PNG.';
}
