// Lightweight viewer checks. The Python validator remains the submission gate.
export function inspectDesign(design, imageWidth, imageHeight) {
    if (!design || !Number.isInteger(design.canvas?.width) || design.canvas.width < 1 ||
        !Number.isInteger(design.canvas?.height) || design.canvas.height < 1 ||
        !Array.isArray(design.elements) || !design.elements.length) {
        throw new Error('Expected an output design with canvas and elements.');
    }
    if (design.canvas.width !== imageWidth || design.canvas.height !== imageHeight) {
        throw new Error('PNG dimensions do not match the JSON canvas. Choose the matching pair.');
    }
    const ids = new Set();
    const warnings = [];
    for (const e of design.elements) {
        if (!e || typeof e.id !== 'string' || ids.has(e.id)) throw new Error('Missing or duplicate element ID.');
        ids.add(e.id);
        if (!['image', 'shape', 'text'].includes(e.type) ||
            !['x', 'y', 'width', 'height'].every(k => Number.isFinite(e[k])) || e.width < 0 || e.height < 0) {
            throw new Error(`Invalid geometry or element type: ${e.id}`);
        }
        if (e.type === 'text') {
            if (typeof e.content !== 'string' || typeof e.font_family !== 'string' ||
                !Number.isFinite(e.font_size) || e.font_size < 1) throw new Error(`Invalid text: ${e.id}`);
            if (e.font_size < 8) warnings.push(`${e.id}: ${e.font_size.toFixed(2)} px — inspect at actual size.`);
        }
    }
    return warnings;
}

export function pairFiles(files) {
    const pairs = new Map();
    for (const file of files) {
        const match = /^(.*)\.(json|png)$/i.exec(file.name);
        if (!match) continue;
        const [, name, extension] = match;
        if (!pairs.has(name)) pairs.set(name, {name});
        const pair = pairs.get(name);
        const key = extension.toLowerCase();
        if (pair[key]) throw new Error(`Duplicate file: ${file.name}`);
        pair[key] = file;
    }
    return [...pairs.values()].sort((a, b) => a.name.localeCompare(b.name));
}
