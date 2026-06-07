export type RuntimeEnv = "development" | "test" | "production";

export function buildSessionCookie(token: string, env: RuntimeEnv): string {
  const secure = env === "production";
  const sameSite = "None";
  const flags = [
    `session=${token}`,
    "Path=/",
    "HttpOnly",
    `SameSite=${sameSite}`,
  ];

  if (secure) {
    flags.push("Secure");
  }

  return flags.join("; ");
}

