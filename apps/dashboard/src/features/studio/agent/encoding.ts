export function encodeBytes(bytes: Uint8Array) {
  let value = ""
  for (const byte of bytes) value += String.fromCharCode(byte)
  return btoa(value)
}

export function decodeBytes(value: string) {
  return Uint8Array.from(atob(value), (char) => char.charCodeAt(0))
}
