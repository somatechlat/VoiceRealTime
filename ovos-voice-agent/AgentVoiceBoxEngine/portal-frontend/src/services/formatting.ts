/**
 * Formatting Utilities
 * Implements Requirements 20.6: Currency formatting according to user locale
 */

export type SupportedCurrency = 'USD' | 'EUR' | 'GBP' | 'JPY' | 'CAD' | 'AUD';

export interface FormatCurrencyOptions {
  currency?: SupportedCurrency;
  locale?: string;
  minimumFractionDigits?: number;
  maximumFractionDigits?: number;
  compact?: boolean;
}

/**
 * Format a number as currency according to locale
 */
export const formatCurrency = (
  amount: number,
  options: FormatCurrencyOptions = {}
): string => {
  const {
    currency = 'USD',
    locale = typeof navigator !== 'undefined' ? navigator.language : 'en-US',
    minimumFractionDigits = 2,
    maximumFractionDigits = 2,
    compact = false,
  } = options;

  const formatter = new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    minimumFractionDigits,
    maximumFractionDigits,
    notation: compact ? 'compact' : 'standard',
  });

  return formatter.format(amount);
};

/**
 * Format a number with locale-specific separators
 */
export const formatNumber = (
  value: number,
  locale?: string,
  options?: Intl.NumberFormatOptions
): string => {
  const resolvedLocale = locale || (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  return new Intl.NumberFormat(resolvedLocale, options).format(value);
};

/**
 * Format a number as a percentage
 */
export const formatPercent = (
  value: number,
  locale?: string,
  decimals: number = 1
): string => {
  const resolvedLocale = locale || (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  return new Intl.NumberFormat(resolvedLocale, {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value / 100);
};

/**
 * Format bytes to human-readable string
 */
export const formatBytes = (bytes: number, decimals: number = 2): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
};

/**
 * Format a date relative to now (e.g., "2 hours ago")
 */
export const formatRelativeTime = (date: Date, locale?: string): string => {
  const resolvedLocale = locale || (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  const rtf = new Intl.RelativeTimeFormat(resolvedLocale, { numeric: 'auto' });

  if (diffDays > 0) return rtf.format(-diffDays, 'day');
  if (diffHours > 0) return rtf.format(-diffHours, 'hour');
  if (diffMins > 0) return rtf.format(-diffMins, 'minute');
  return rtf.format(-diffSecs, 'second');
};

/**
 * Format a date according to locale
 */
export const formatDate = (
  date: Date,
  locale?: string,
  options?: Intl.DateTimeFormatOptions
): string => {
  const resolvedLocale = locale || (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  };
  return new Intl.DateTimeFormat(resolvedLocale, options || defaultOptions).format(date);
};

/**
 * Format a date and time according to locale
 */
export const formatDateTime = (
  date: Date,
  locale?: string,
  options?: Intl.DateTimeFormatOptions
): string => {
  const resolvedLocale = locale || (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  };
  return new Intl.DateTimeFormat(resolvedLocale, options || defaultOptions).format(date);
};

/**
 * Get currency symbol for a currency code
 */
export const getCurrencySymbol = (currency: SupportedCurrency, locale?: string): string => {
  const resolvedLocale = locale || (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  return new Intl.NumberFormat(resolvedLocale, {
    style: 'currency',
    currency,
    currencyDisplay: 'narrowSymbol',
  })
    .formatToParts(0)
    .find(part => part.type === 'currency')?.value || currency;
};
