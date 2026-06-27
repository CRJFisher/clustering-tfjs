import { REPO_URL, build_repo_url } from "./repo_links";

describe("build_repo_url", () => {
  test("tags each surface with source, medium, and campaign", () => {
    const url = new URL(build_repo_url(REPO_URL, "header"));
    expect(url.searchParams.get("utm_source")).toBe("demo");
    expect(url.searchParams.get("utm_medium")).toBe("header");
    expect(url.searchParams.get("utm_campaign")).toBe("clustering_demo");
  });

  test("the medium reflects the link surface", () => {
    expect(
      new URL(build_repo_url(REPO_URL, "code_panel")).searchParams.get(
        "utm_medium",
      ),
    ).toBe("code_panel");
    expect(
      new URL(build_repo_url(REPO_URL, "footer")).searchParams.get(
        "utm_medium",
      ),
    ).toBe("footer");
  });

  test("preserves a query already on the base URL", () => {
    const url = new URL(build_repo_url("https://example.com/x?ref=a", "footer"));
    expect(url.searchParams.get("ref")).toBe("a");
    expect(url.searchParams.get("utm_medium")).toBe("footer");
  });

  test("re-tagging overwrites the medium without duplicating keys", () => {
    const once = build_repo_url(REPO_URL, "header");
    const twice = build_repo_url(once, "footer");
    const url = new URL(twice);
    expect(url.searchParams.getAll("utm_medium")).toEqual(["footer"]);
  });
});
