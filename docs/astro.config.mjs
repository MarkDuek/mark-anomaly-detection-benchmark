import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import mdx from "@astrojs/mdx";
import starlight from "@astrojs/starlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

export default defineConfig({
  integrations: [
    starlight({
      title: "Multivariate Anomaly Detection",
      description:
        "A benchmark-oriented workspace for multivariate time-series anomaly detection.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/MarkDuek/mark-anomaly-detection-benchmark",
        },
      ],
      sidebar: [
        { label: "Overview", link: "/" },
        { label: "Models", link: "/models/" },
        { label: "Protocols", link: "/protocols/" },
        { label: "Thresholds", link: "/thresholds/" },
        { label: "Datasets", link: "/datasets/" },
        { label: "Metrics", link: "/metrics/" },
        {
          label: "Runs",
          items: [
            { label: "Compare runs", link: "/runs/" },
            { label: "Run explorer", link: "/runs/explorer/" },
          ],
        },
      ],
      customCss: ["./src/styles/custom.css"],
    }),
    mdx(),
  ],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
});
