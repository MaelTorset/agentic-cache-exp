export const qrRoute = {
  path: "/join/scan",
  cameraPermission: "required",
  fallbackPath: "/join/manual-code",
};

export function shouldOpenManualFallback(errorName: string): boolean {
  return errorName === "NotAllowedError" || errorName === "NotFoundError";
}

