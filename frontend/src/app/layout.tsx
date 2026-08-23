import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StocKast — Framework dự báo chuỗi thời gian cổ phiếu",
  description: "Framework Data Scientist và ML Engineer kết hợp ML truyền thống với RNN để nghiên cứu dự báo chuỗi thời gian cổ phiếu.",
};

export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="vi"><body>{children}</body></html>;
}
