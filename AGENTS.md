# Contributor guidance

Keep backend selection and entered credentials session-scoped. Hosted deployments
require the caller's Comfy key; never bundle an owner key or set HFR_PRIVATE_HOSTED.
Do not print, save or return credentials in logs, workflow graphs or recipe exports.
Keep the app and ComfyUI-HeroFrameRig nodes on the same hfr_core implementation.
Preserve recipe replay and edge_cleanup=0 compatibility. Run focused CPU tests
for renderer changes; paid jobs, model downloads and publication require the
maintainer's authorization. Keep user media out of source commits.
The app code is Apache-2.0 and the focus pack MIT; model and media licences are separate.
