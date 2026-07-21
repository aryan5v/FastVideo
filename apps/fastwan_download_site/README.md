# FastWan QAD downloads

Railway-hosted landing page and download broker for the FastWan QAD macOS app.
Release binaries live in a private Railway S3-compatible bucket. Stable routes
such as `/download/ema` create a short-lived presigned bucket URL and redirect
the native installer without proxying multi-gigabyte downloads through the web
service.

## Local development

```console
npm install
npm test
npm start
```

Without bucket credentials, the page and catalog work but every unpublished
download fails closed with HTTP 503.

## Railway variables

Connect a Railway Storage Bucket and expose its S3 credentials using either the
`BUCKET_*` names or Railway's standard `AWS_*` names. Set `PUBLIC_BASE_URL` to
the final HTTPS domain so `/catalog.json` emits stable installer URLs.

## Upload an artifact

```console
npm run upload -- app /path/to/FastWan-QAD-macOS.zip
```

The multipart uploader calculates SHA-256, uploads the object, and updates
`artifacts.json`. Commit the resulting manifest and copy the model entries from
the deployed `/catalog.json` into the app's bundled model catalog before the
final signed build.
