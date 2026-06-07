export function buildInvoiceNumber(customerId: string, sequence: number): string {
  return `INV-${customerId}-${String(sequence).padStart(5, "0")}`;
}

export function computeInvoiceTotal(lines: Array<{ quantity: number; unitPrice: number }>): number {
  return lines.reduce((total, line) => total + line.quantity * line.unitPrice, 0);
}

