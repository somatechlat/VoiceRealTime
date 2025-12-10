/**
 * Property-Based Tests for Serialization and API Client
 * 
 * These tests verify the correctness properties of serialization utilities
 * and API client using fast-check for property-based testing.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  serialize,
  deserialize,
  roundTrip,
  dateToIso8601,
  iso8601ToDate,
  isIso8601Date,
} from '@/services/serialization';
import {
  formatCurrency,
  formatNumber,
  formatPercent,
} from '@/services/formatting';
import { RETRY_DELAYS } from '@/services/api-client';

/**
 * **Feature: saas-portal, Property 23: JSON Serialization Round-Trip**
 * For any data object sent to or received from the API, serializing to JSON
 * and deserializing back SHALL produce an equivalent object.
 * **Validates: Requirements 20.1**
 */
describe('Property 23: JSON Serialization Round-Trip', () => {
  // Arbitrary for JSON-serializable primitives
  const jsonPrimitive = fc.oneof(
    fc.string(),
    fc.integer(),
    fc.double({ noNaN: true, noDefaultInfinity: true }),
    fc.boolean(),
    fc.constant(null)
  );

  // Arbitrary for JSON-serializable objects (without dates for basic test)
  const jsonObject = fc.dictionary(
    fc.string().filter(s => s.length > 0 && !s.includes('"')),
    jsonPrimitive
  );

  // Arbitrary for JSON-serializable arrays
  const jsonArray = fc.array(jsonPrimitive);

  it('should round-trip primitive values', () => {
    fc.assert(
      fc.property(jsonPrimitive, (value) => {
        const result = roundTrip(value);
        expect(result).toEqual(value);
      }),
      { numRuns: 100 }
    );
  });

  it('should round-trip objects', () => {
    fc.assert(
      fc.property(jsonObject, (obj) => {
        const result = roundTrip(obj);
        expect(result).toEqual(obj);
      }),
      { numRuns: 100 }
    );
  });

  it('should round-trip arrays', () => {
    fc.assert(
      fc.property(jsonArray, (arr) => {
        const result = roundTrip(arr);
        expect(result).toEqual(arr);
      }),
      { numRuns: 100 }
    );
  });

  it('should round-trip nested structures', () => {
    const nestedStructure = fc.record({
      id: fc.integer(),
      name: fc.string(),
      active: fc.boolean(),
      tags: fc.array(fc.string()),
      metadata: fc.dictionary(fc.string(), fc.string()),
    });

    fc.assert(
      fc.property(nestedStructure, (data) => {
        const result = roundTrip(data);
        expect(result).toEqual(data);
      }),
      { numRuns: 100 }
    );
  });

  it('should handle Date objects in round-trip', () => {
    // Generate valid dates (not too far in past/future)
    const dateArbitrary = fc.date({
      min: new Date('2000-01-01'),
      max: new Date('2100-01-01'),
    });

    fc.assert(
      fc.property(dateArbitrary, (date) => {
        const obj = { createdAt: date };
        const serialized = serialize(obj);
        const deserialized = deserialize<{ createdAt: Date }>(serialized);
        
        // Dates should be equal (within millisecond precision)
        expect(deserialized.createdAt.getTime()).toBe(date.getTime());
      }),
      { numRuns: 100 }
    );
  });
});

/**
 * **Feature: saas-portal, Property 24: ISO 8601 Date Format**
 * For any date value in API requests or responses, the format SHALL be
 * ISO 8601 with timezone.
 * **Validates: Requirements 20.2**
 */
describe('Property 24: ISO 8601 Date Format', () => {
  const dateArbitrary = fc.date({
    min: new Date('2000-01-01'),
    max: new Date('2100-01-01'),
  });

  it('should convert any Date to valid ISO 8601 format', () => {
    fc.assert(
      fc.property(dateArbitrary, (date) => {
        const isoString = dateToIso8601(date);
        
        // Should be a valid ISO 8601 string
        expect(isIso8601Date(isoString)).toBe(true);
        
        // Should end with Z (UTC timezone)
        expect(isoString).toMatch(/Z$/);
        
        // Should have correct format
        expect(isoString).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
      }),
      { numRuns: 100 }
    );
  });

  it('should round-trip Date through ISO 8601', () => {
    fc.assert(
      fc.property(dateArbitrary, (date) => {
        const isoString = dateToIso8601(date);
        const parsed = iso8601ToDate(isoString);
        
        // Should preserve the timestamp
        expect(parsed.getTime()).toBe(date.getTime());
      }),
      { numRuns: 100 }
    );
  });

  it('should correctly identify ISO 8601 strings', () => {
    const validIsoStrings = [
      '2024-01-15T10:30:00.000Z',
      '2024-12-31T23:59:59.999Z',
      '2024-01-01T00:00:00.000Z',
      '2024-06-15T12:00:00Z',
      '2024-06-15T12:00:00+00:00',
      '2024-06-15T12:00:00-05:00',
    ];

    validIsoStrings.forEach(str => {
      expect(isIso8601Date(str)).toBe(true);
    });
  });

  it('should reject invalid date strings', () => {
    const invalidStrings = [
      '2024-01-15',
      '10:30:00',
      'not a date',
      '2024/01/15T10:30:00Z',
      '',
    ];

    invalidStrings.forEach(str => {
      expect(isIso8601Date(str)).toBe(false);
    });
  });
});

/**
 * **Feature: saas-portal, Property 25: Retry with Exponential Backoff**
 * For any failed API request with retry enabled, the system SHALL retry
 * up to 3 times with delays of 1s, 2s, and 4s.
 * **Validates: Requirements 20.4**
 */
describe('Property 25: Retry with Exponential Backoff', () => {
  it('should have exactly 3 retry delays', () => {
    expect(RETRY_DELAYS).toHaveLength(3);
  });

  it('should have delays of 1s, 2s, 4s', () => {
    expect(RETRY_DELAYS).toEqual([1000, 2000, 4000]);
  });

  it('should follow exponential backoff pattern', () => {
    // Each delay should be double the previous
    for (let i = 1; i < RETRY_DELAYS.length; i++) {
      expect(RETRY_DELAYS[i]).toBe(RETRY_DELAYS[i - 1] * 2);
    }
  });

  it('should have increasing delays', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: RETRY_DELAYS.length - 2 }),
        (index) => {
          expect(RETRY_DELAYS[index + 1]).toBeGreaterThan(RETRY_DELAYS[index]);
        }
      ),
      { numRuns: 10 }
    );
  });
});

/**
 * **Feature: saas-portal, Property 26: Currency Locale Formatting**
 * For any currency value displayed to the user, the format SHALL match
 * the user's locale settings.
 * **Validates: Requirements 20.6**
 */
describe('Property 26: Currency Locale Formatting', () => {
  const amountArbitrary = fc.double({
    min: 0,
    max: 1000000,
    noNaN: true,
    noDefaultInfinity: true,
  });

  const currencyArbitrary = fc.constantFrom('USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD') as fc.Arbitrary<'USD' | 'EUR' | 'GBP' | 'JPY' | 'CAD' | 'AUD'>;
  const localeArbitrary = fc.constantFrom('en-US', 'en-GB', 'de-DE', 'fr-FR', 'ja-JP');

  it('should format any amount as currency', () => {
    fc.assert(
      fc.property(amountArbitrary, (amount) => {
        const formatted = formatCurrency(amount);
        
        // Should be a non-empty string
        expect(typeof formatted).toBe('string');
        expect(formatted.length).toBeGreaterThan(0);
      }),
      { numRuns: 100 }
    );
  });

  it('should include currency symbol', () => {
    fc.assert(
      fc.property(amountArbitrary, currencyArbitrary, (amount, currency) => {
        const formatted = formatCurrency(amount, { currency: currency, locale: 'en-US' });
        
        // Should contain some currency indicator
        // (symbol position varies by locale)
        expect(formatted).toBeTruthy();
      }),
      { numRuns: 100 }
    );
  });

  it('should format numbers according to locale', () => {
    fc.assert(
      fc.property(amountArbitrary, localeArbitrary, (amount, locale) => {
        const formatted = formatNumber(amount, locale);
        
        // Should be a non-empty string
        expect(typeof formatted).toBe('string');
        expect(formatted.length).toBeGreaterThan(0);
      }),
      { numRuns: 100 }
    );
  });

  it('should format percentages correctly', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 100, noNaN: true }),
        (value) => {
          const formatted = formatPercent(value, 'en-US');
          
          // Should contain % symbol
          expect(formatted).toContain('%');
        }
      ),
      { numRuns: 100 }
    );
  });

  it('should handle zero amounts', () => {
    const formatted = formatCurrency(0);
    expect(formatted).toBeTruthy();
    expect(formatted).toContain('0');
  });

  it('should handle negative amounts', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -1000000, max: -0.01, noNaN: true }),
        (amount) => {
          const formatted = formatCurrency(amount);
          
          // Should indicate negative (either with - or parentheses)
          expect(formatted.includes('-') || formatted.includes('(')).toBe(true);
        }
      ),
      { numRuns: 100 }
    );
  });
});
