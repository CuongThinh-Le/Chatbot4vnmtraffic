export function isMetaQuestion(text: string): boolean {
  if (!text) return false;

  const normalized = text
    .toLowerCase()
    .trim()
    .replace(/[?.!]/g, "");

  const metaPatterns = [
    // xác nhận / nghi ngờ
    "bạn chắc chưa",
    "bạn có chắc",
    "có đúng không",
    "đúng không",
    "xác nhận lại",
    "nói rõ hơn",
    "giải thích thêm",

    // tham chiếu câu trước
    "cái trên",
    "như trên",
    "ở trên",
    "ý trên",
    "trường hợp trên",
    "vậy còn"
  ];

  return metaPatterns.some((pattern) =>
    normalized.includes(pattern)
  );
}
