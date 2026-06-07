import { buildSessionCookie } from "../src/auth/cookie";

describe("buildSessionCookie", () => {
  it("uses a localhost-compatible cookie in development", () => {
    const cookie = buildSessionCookie("dev-token", "development");

    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("SameSite=Lax");
    expect(cookie).not.toContain("Secure");
  });

  it("keeps cross-site secure cookies in production", () => {
    const cookie = buildSessionCookie("prod-token", "production");

    expect(cookie).toContain("SameSite=None");
    expect(cookie).toContain("Secure");
  });
});

