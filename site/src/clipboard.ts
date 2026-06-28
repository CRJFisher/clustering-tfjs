// Copies text to the clipboard, resolving to whether it succeeded. The Clipboard
// API is absent outside a secure context and can reject without permission, so
// every caller gets a boolean to drive an honest "copied" vs "copy it manually"
// affordance rather than an unhandled rejection.
export function copy_text(text: string): Promise<boolean> {
  if (!navigator.clipboard) return Promise.resolve(false);
  return navigator.clipboard.writeText(text).then(
    () => true,
    () => false,
  );
}
