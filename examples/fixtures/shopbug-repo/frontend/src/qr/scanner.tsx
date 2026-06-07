export function QrScannerPanel() {
  return (
    <section>
      <h1>Scan QR invite</h1>
      <button type="button">Start camera</button>
    </section>
  );
}

export function parseQrPayload(payload: string): { inviteId: string } {
  const parsed = new URL(payload);
  return { inviteId: parsed.searchParams.get("invite") ?? "" };
}

