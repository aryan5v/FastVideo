// Pages Function for /api/* (info, ballot, vote, results, exports). Logic lives in ../../src/app.js.
import bundle from "../../src/bundle_data.js";
import { handleApi } from "../../src/app.js";

export const onRequest = ({ request, env }) => handleApi(request, env, bundle);
