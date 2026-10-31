import {test} from 'node:test';
import assert from 'node:assert/strict';
import {inspectDesign, pairFiles} from '../src/review.js';
const design = () => ({canvas: {width: 728, height: 90}, elements: [{id: 'copy', type: 'text', x: 0, y: 0, width: 100, height: 20, content: '<script>hello</script>', font_family: 'Solstice Sans', font_size: 7}]});
test('flags tiny text without rewriting copy', () => {
    const d = design(), before = JSON.stringify(d);
    assert.equal(inspectDesign(d, 728, 90).length, 1);
    assert.equal(JSON.stringify(d), before);
});
test('rejects mismatched PNG, duplicate IDs and malformed text', () => {
    assert.throws(() => inspectDesign(design(), 100, 100), /dimensions/);
    const d = design(); d.elements.push({...d.elements[0]});
    assert.throws(() => inspectDesign(d, 728, 90), /duplicate/);
    const invalid = design(); invalid.elements[0].font_size = null;
    assert.throws(() => inspectDesign(invalid, 728, 90), /Invalid text/);
});
test('pairs exact basenames, exposes missing files and rejects duplicates', () => {
    const files = [{name:'a_output.json'}, {name:'b_output.png'}, {name:'a_output.png'}];
    const pairs = pairFiles(files);
    assert.equal(pairs[0].json, files[0]); assert.equal(pairs[0].png, files[2]);
    assert.equal(pairs[1].json, undefined);
    assert.throws(() => pairFiles([files[0], files[0]]), /Duplicate/);
});
