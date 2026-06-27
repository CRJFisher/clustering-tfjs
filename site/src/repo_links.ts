// Every repo/demo link is UTM-tagged so GitHub stars and npm installs that start
// at the demo are attributable to the surface that drove them. One builder is the
// single source of the tag scheme, so no link can ship untagged or mistyped.

// The surface a link lives on — becomes utm_medium.
export type UtmMedium = "header" | "code_panel" | "footer";

export const REPO_URL = "https://github.com/CRJFisher/clustering-tfjs";

const UTM_SOURCE = "demo";
const UTM_CAMPAIGN = "clustering_demo";

// Appends the utm_* params without clobbering any query already on the base URL.
export function build_repo_url(base_url: string, medium: UtmMedium): string {
  const url = new URL(base_url);
  url.searchParams.set("utm_source", UTM_SOURCE);
  url.searchParams.set("utm_medium", medium);
  url.searchParams.set("utm_campaign", UTM_CAMPAIGN);
  return url.toString();
}
