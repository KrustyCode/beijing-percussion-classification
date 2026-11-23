// constants/errors.ts
export type ErrorCode =
  | "UNSUPPORTED_FORMAT"
  | "CORRUPT_FILE"
  | "FILE_TOO_LARGE"
  | "DURATION_TOO_LONG"
  | "DURATION_TOO_SHORT"
  | "SILENT_AUDIO"
  | "UNPROCESSABLE_AUDIO"
  | "NETWORK_ERROR"
  | "UNKNOWN_ERROR";

export interface ErrorInfo {
  title: string; // short heading, e.g. "Corrupt file"
  message: string; // what went wrong
  action: string; // what the user should do next
}

// Validation limits: one place for the numbers (FR-03).
// Must match backend/utils/validation.py (MAX_SIZE, MIN/MAX_UPLOAD_DURATION).
export const LIMITS = {
  formats: ["WAV", "MP3"],
  maxSizeMb: 10,
  minDurationS: 0.05,
  maxDurationS: 10,
} as const;

// English number format: 0.5 -> "0.5"
export const fmt = (n: number) => n.toLocaleString("en-US");

export const ERRORS: Record<ErrorCode, ErrorInfo> = {
  UNSUPPORTED_FORMAT: {
    title: "Unsupported format",
    message: `The file must be ${LIMITS.formats.join(" or ")}.`,
    action: `Convert the file to ${LIMITS.formats.join(" or ")}, then upload it again.`,
  },
  CORRUPT_FILE: {
    title: "Corrupt file",
    message: "The file could not be read as audio.",
    action: "Make sure the file plays, then upload it again.",
  },
  FILE_TOO_LARGE: {
    title: "File too large",
    message: `The file is larger than the ${fmt(LIMITS.maxSizeMb)} MB limit.`,
    action: "Trim or compress the recording, then upload it again.",
  },
  DURATION_TOO_LONG: {
    title: "Recording too long",
    message: `The recording is longer than the ${fmt(LIMITS.maxDurationS)} second limit.`,
    action: "Trim the recording, then upload it again.",
  },
  DURATION_TOO_SHORT: {
    title: "Recording too short",
    message: `The recording is shorter than ${fmt(LIMITS.minDurationS)} seconds.`,
    action: "Upload a recording that contains the whole stroke.",
  },
  SILENT_AUDIO: {
    title: "Only silence",
    message: "No sound was detected in the file.",
    action: "Make sure the instrument stroke can be heard in the recording.",
  },
  UNPROCESSABLE_AUDIO: {
    title: "Audio could not be analyzed",
    message: "Audio features could not be computed from this file.",
    action: "Upload another recording with a clear stroke.",
  },
  NETWORK_ERROR: {
    title: "Could not connect",
    message: "The request did not reach the server.",
    action: "Check your internet connection, then try again.",
  },
  UNKNOWN_ERROR: {
    title: "Something went wrong",
    message: "The file could not be processed.",
    action: "Try again in a moment.",
  },
};

export function isErrorCode(value: unknown): value is ErrorCode {
  return typeof value === "string" && value in ERRORS;
}

// Always returns something displayable, even for unexpected codes
export function getError(code: unknown): ErrorInfo {
  return isErrorCode(code) ? ERRORS[code] : ERRORS.UNKNOWN_ERROR;
}
