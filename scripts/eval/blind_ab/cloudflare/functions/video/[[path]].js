// Pages Function for /video/<ballot_id>/<left|right>: streams the blinded video (Range requests supported).
import bundle from "../../src/bundle_data.js";
import { handleVideo } from "../../src/app.js";

export const onRequest = ({ request, env }) => handleVideo(request, env, bundle);
