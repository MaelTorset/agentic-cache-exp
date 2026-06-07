export const analyticsEvents = {
  qrCameraStarted: "qr_camera_started",
  qrCameraDenied: "qr_camera_denied",
  loginSucceeded: "login_succeeded",
  invoiceDownloaded: "invoice_downloaded",
};

export function buildEventPayload(name: string, userId: string): Record<string, string> {
  return { name, userId };
}

