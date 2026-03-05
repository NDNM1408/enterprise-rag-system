import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// No authentication required for demo
export function middleware(request: NextRequest) {
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
