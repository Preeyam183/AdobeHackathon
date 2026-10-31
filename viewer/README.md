# CHAMELEON Viewer

Adobe Express panel and standalone viewer for the actual engine output. Select
matching `*_output.json` and `*_output.png` files together. Multiple pairs are
supported. Review the PNG at actual size, inspect every text string and font size,
and download either artifact. Nothing is uploaded by file selection.

In Adobe Express, **Add to Express** inserts the original rendered PNG into the
current page as one image. It does not recreate editable design elements, run
the Python engine in the browser, or regenerate image assets. The engine remains
a separate offline container. Browser checks are lightweight; run the Python
preflight for full schema/contract validation.

## Standalone preview (no Node dependencies or internet required)

From the repository root:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory viewer/src
```

Open http://127.0.0.1:8765 and select the engine outputs. The standalone page does
not load Adobe's SDK. Use actual-size mode to inspect narrow banners; the preview
scrolls horizontally instead of hiding unreadable text with magnification.

## Adobe Express development and packaging

Use a current Node LTS version supported by Adobe's tooling, then:

```sh
cd viewer
npm ci
npm test
npm run build
npm run start
```

Adobe's CLI serves the add-on over HTTPS. Follow its certificate prompts and
open the supplied Adobe Express development link. Enable Add-on Development in
your Adobe account, open a document, and connect to the local development server.
Load matching output files and click **Add to Express**. Verify the inserted image
in the actual Adobe host before submission; the automated insertion test uses a
mock SDK and cannot establish real-host compatibility or organizer acceptance.

```sh
npm run package
```

The CLI validates the manifest and creates `dist.zip`. Submit that bundle when a
viewer is requested. `node_modules` is build tooling and is not shipped in the
bundle or the Python container. The pinned Adobe CLI currently has dependency
security advisories; they concern development tools, not runtime dependencies
of this static viewer. Do not expose the development server publicly.

References: [manifest v2](https://developer.adobe.com/express/add-ons/docs/references/manifest/),
[local development](https://developer.adobe.com/express/add-ons/docs/guides/getting-started/local-development/dev-tooling),
[image insertion](https://developer.adobe.com/express/add-ons/docs/references/addonsdk/app-document/).
