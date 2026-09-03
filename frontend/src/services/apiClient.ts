/**
 * API Client wrapper cho giao tiếp với Backend
 * Mặc định trỏ đến http://localhost:8000 theo BACKEND.md
 */

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";

interface RequestOptions {
  timeoutMs?: number;
  headers?: Record<string, string>;
}

export class ApiClient {
  private static baseUrl = API_BASE_URL;

  public static setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  public static getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * GET Request
   */
  public static async get<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 10000);

    try {
      const response = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
          ...options.headers,
        },
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (err: any) {
      clearTimeout(timeout);
      throw err;
    }
  }

  /**
   * POST Request (JSON or FormData)
   */
  public static async post<T>(endpoint: string, data: any, options: RequestOptions = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 30000);

    const isFormData = typeof FormData !== "undefined" && data instanceof FormData;
    const headers: Record<string, string> = {
      Accept: "application/json",
      ...options.headers,
    };

    if (!isFormData) {
      headers["Content-Type"] = "application/json";
    }

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: isFormData ? data : JSON.stringify(data),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (err: any) {
      clearTimeout(timeout);
      throw err;
    }
  }
}
