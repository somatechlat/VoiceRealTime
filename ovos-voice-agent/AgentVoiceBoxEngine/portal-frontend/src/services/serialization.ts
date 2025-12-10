/**
 * JSON Serialization Utilities
 * Implements Requirements 20.1, 20.2, 20.3: JSON serialization with ISO 8601 dates
 */

import { z } from 'zod';

/**
 * ISO 8601 date regex pattern
 */
const ISO_8601_REGEX = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?(Z|[+-]\d{2}:\d{2})$/;

/**
 * Check if a string is an ISO 8601 date
 */
export const isIso8601Date = (value: string): boolean => {
  return ISO_8601_REGEX.test(value);
};

/**
 * Convert a Date to ISO 8601 string with timezone
 */
export const dateToIso8601 = (date: Date): string => {
  return date.toISOString();
};

/**
 * Parse an ISO 8601 string to Date
 */
export const iso8601ToDate = (isoString: string): Date => {
  const date = new Date(isoString);
  if (isNaN(date.getTime())) {
    throw new Error(`Invalid ISO 8601 date: ${isoString}`);
  }
  return date;
};

/**
 * Custom JSON replacer that handles Date objects
 */
export const jsonReplacer = (_key: string, value: unknown): unknown => {
  if (value instanceof Date) {
    return dateToIso8601(value);
  }
  return value;
};

/**
 * Custom JSON reviver that converts ISO 8601 strings to Date objects
 */
export const jsonReviver = (_key: string, value: unknown): unknown => {
  if (typeof value === 'string' && isIso8601Date(value)) {
    return iso8601ToDate(value);
  }
  return value;
};

/**
 * Serialize an object to JSON string with Date handling
 */
export const serialize = <T>(data: T): string => {
  return JSON.stringify(data, jsonReplacer);
};

/**
 * Deserialize a JSON string with Date handling
 */
export const deserialize = <T>(json: string): T => {
  return JSON.parse(json, jsonReviver) as T;
};

/**
 * Serialize and deserialize (round-trip) - useful for testing
 */
export const roundTrip = <T>(data: T): T => {
  return deserialize<T>(serialize(data));
};

/**
 * Validate data against a Zod schema after deserialization
 */
export const deserializeAndValidate = <T>(
  json: string,
  schema: z.ZodSchema<T>
): T => {
  const data = deserialize<unknown>(json);
  return schema.parse(data);
};

/**
 * Safe deserialize that returns null on error
 */
export const safeDeserialize = <T>(json: string): T | null => {
  try {
    return deserialize<T>(json);
  } catch {
    return null;
  }
};

/**
 * Deep clone an object using serialization
 */
export const deepClone = <T>(data: T): T => {
  return roundTrip(data);
};

// Common Zod schemas for API responses
export const dateSchema = z.string().refine(isIso8601Date, {
  message: 'Invalid ISO 8601 date format',
}).transform(iso8601ToDate);

export const paginatedResponseSchema = <T extends z.ZodTypeAny>(itemSchema: T) =>
  z.object({
    data: z.array(itemSchema),
    cursor: z.string().nullable(),
    hasMore: z.boolean(),
    total: z.number().optional(),
  });
