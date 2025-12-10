/**
 * Layout Components Unit Tests
 * Tests for Sidebar, Header, DashboardLayout, and MobileNav
 * Implements Task 11: Customer Portal Layout & Navigation
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { MobileNav } from "@/components/layout/MobileNav";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/dashboard"),
  useRouter: vi.fn(() => ({
    push: vi.fn(),
    replace: vi.fn(),
  })),
}));

// Mock AuthContext
const mockUser = {
  id: "user-123",
  tenantId: "tenant-456",
  email: "test@example.com",
  username: "testuser",
  roles: ["owner"],
  permissions: ["api_keys:view", "billing:view", "team:view", "usage:view", "settings:manage"],
};

const mockAuthContext = {
  user: mockUser,
  isLoading: false,
  isAuthenticated: true,
  login: vi.fn(),
  logout: vi.fn(),
  refreshSession: vi.fn(),
  hasPermission: vi.fn((permission: string) => mockUser.permissions.includes(permission)),
  hasRole: vi.fn((role: string) => mockUser.roles.includes(role)),
  isAdmin: vi.fn(() => true),
  isPlatformAdmin: vi.fn(() => false),
};

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => mockAuthContext,
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock ThemeContext
vi.mock("@/contexts/ThemeContext", () => ({
  useTheme: () => ({
    theme: "dark",
    resolvedTheme: "dark",
    setTheme: vi.fn(),
  }),
}));

describe("Sidebar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders logo and brand name", () => {
    render(<Sidebar />);
    expect(screen.getByText("AgentVoiceBox")).toBeInTheDocument();
  });

  it("renders all navigation items for owner role", () => {
    render(<Sidebar />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("API Keys")).toBeInTheDocument();
    expect(screen.getByText("Usage")).toBeInTheDocument();
    expect(screen.getByText("Billing")).toBeInTheDocument();
    expect(screen.getByText("Team")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("renders documentation link", () => {
    render(<Sidebar />);
    const docLink = screen.getByText("Documentation");
    expect(docLink).toBeInTheDocument();
    expect(docLink.closest("a")).toHaveAttribute("href", "https://docs.agentvoicebox.com");
    expect(docLink.closest("a")).toHaveAttribute("target", "_blank");
  });

  it("renders user info section", () => {
    render(<Sidebar />);
    expect(screen.getByText("testuser")).toBeInTheDocument();
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });

  it("highlights active navigation item", () => {
    render(<Sidebar />);
    const dashboardLink = screen.getByText("Dashboard").closest("a");
    expect(dashboardLink).toHaveAttribute("aria-current", "page");
  });

  it("calls onClose when navigation item is clicked", () => {
    const onClose = vi.fn();
    render(<Sidebar onClose={onClose} />);
    
    fireEvent.click(screen.getByText("API Keys"));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls logout when sign out button is clicked", async () => {
    render(<Sidebar />);
    
    const logoutButton = screen.getByLabelText("Sign out");
    fireEvent.click(logoutButton);
    
    expect(mockAuthContext.logout).toHaveBeenCalled();
  });

  it("hides navigation items user lacks permission for", () => {
    // Override hasPermission to deny billing:view
    mockAuthContext.hasPermission.mockImplementation((permission: string) => {
      return permission !== "billing:view";
    });

    render(<Sidebar />);
    
    // Billing should not be visible
    expect(screen.queryByText("Billing")).not.toBeInTheDocument();
    
    // Reset mock
    mockAuthContext.hasPermission.mockImplementation((permission: string) => 
      mockUser.permissions.includes(permission)
    );
  });

  it("renders theme toggle", () => {
    render(<Sidebar />);
    expect(screen.getByText("Theme")).toBeInTheDocument();
  });

  it("has proper accessibility attributes", () => {
    render(<Sidebar />);
    
    const nav = screen.getByRole("navigation", { name: "Main navigation" });
    expect(nav).toBeInTheDocument();
    
    const primaryNav = screen.getByRole("navigation", { name: "Primary" });
    expect(primaryNav).toBeInTheDocument();
    
    const secondaryNav = screen.getByRole("navigation", { name: "Secondary" });
    expect(secondaryNav).toBeInTheDocument();
  });
});

describe("Header", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders page title", () => {
    render(<Header title="Dashboard" />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("renders page description when provided", () => {
    render(<Header title="Dashboard" description="Overview of your account" />);
    expect(screen.getByText("Overview of your account")).toBeInTheDocument();
  });

  it("renders current date", () => {
    render(<Header title="Dashboard" />);
    // Date should be in format like "Monday, January 1, 2024"
    const today = new Date();
    const expectedDay = today.toLocaleDateString('en-US', { weekday: 'long' });
    expect(screen.getByText(new RegExp(expectedDay))).toBeInTheDocument();
  });

  it("renders search input on desktop", () => {
    render(<Header title="Dashboard" />);
    expect(screen.getByLabelText("Search")).toBeInTheDocument();
  });

  it("renders notifications button", () => {
    render(<Header title="Dashboard" />);
    expect(screen.getByLabelText("View notifications")).toBeInTheDocument();
  });

  it("renders user avatar", () => {
    render(<Header title="Dashboard" />);
    // Avatar should show first letter of username
    expect(screen.getByText("T")).toBeInTheDocument();
  });

  it("calls onMenuClick when menu button is clicked", () => {
    const onMenuClick = vi.fn();
    render(<Header title="Dashboard" onMenuClick={onMenuClick} />);
    
    const menuButton = screen.getByLabelText("Open menu");
    fireEvent.click(menuButton);
    
    expect(onMenuClick).toHaveBeenCalled();
  });

  it("shows close icon when mobile menu is open", () => {
    render(<Header title="Dashboard" isMobileMenuOpen={true} />);
    expect(screen.getByLabelText("Close menu")).toBeInTheDocument();
  });

  it("shows tenant ID badge", () => {
    render(<Header title="Dashboard" />);
    expect(screen.getByText("tenant-4...")).toBeInTheDocument();
  });
});

describe("MobileNav", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAuthContext.hasPermission.mockImplementation((permission: string) => 
      mockUser.permissions.includes(permission)
    );
  });

  it("renders navigation items", () => {
    render(<MobileNav />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("API Keys")).toBeInTheDocument();
    expect(screen.getByText("Billing")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByText("More")).toBeInTheDocument();
  });

  it("highlights active navigation item", () => {
    render(<MobileNav />);
    const dashboardLink = screen.getByText("Dashboard").closest("a");
    expect(dashboardLink).toHaveAttribute("aria-current", "page");
  });

  it("calls onMoreClick when More button is clicked", () => {
    const onMoreClick = vi.fn();
    render(<MobileNav onMoreClick={onMoreClick} />);
    
    fireEvent.click(screen.getByText("More"));
    expect(onMoreClick).toHaveBeenCalled();
  });

  it("hides items user lacks permission for", () => {
    mockAuthContext.hasPermission.mockImplementation((permission: string) => {
      return permission !== "billing:view";
    });

    render(<MobileNav />);
    expect(screen.queryByText("Billing")).not.toBeInTheDocument();
  });

  it("has proper accessibility attributes", () => {
    render(<MobileNav />);
    
    const nav = screen.getByRole("navigation", { name: "Mobile navigation" });
    expect(nav).toBeInTheDocument();
  });

  it("renders More button with proper aria-label", () => {
    render(<MobileNav />);
    expect(screen.getByLabelText("More options")).toBeInTheDocument();
  });
});
