export function isComparisonQuestion(text: string): boolean {
  const patterns = [
    "khác nhau",
    "phân biệt",
    "so sánh",
    "so với",
    "khác gì",
    "hai khái niệm",
    "điểm khác",
    "giống và khác",
    "trên khác",
  ];

  const lower = text.toLowerCase();
  return patterns.some(p => lower.includes(p));
}
