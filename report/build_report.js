// Builds report/Group 1 - Project A Report.docx
// All figures come from code: the notebook's plot outputs (extracted to report/figures/cellNNN_k.png)
// and scripts/report_figures.py (Part 2 charts). Run: node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, LevelFormat, WidthType, ShadingType, BorderStyle, PageBreak, Footer, PageNumber,
  TableOfContents,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const FONT = "Calibri";
const TEXT_W = 9026; // A4 width minus 2 x 1440 margins, in DXA
const ACCENT = "1F4E79";

// ---------- helpers ----------
// "**bold** text" -> runs
function runs(text, opts = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) =>
    part.startsWith("**")
      ? new TextRun({ text: part.slice(2, -2), bold: true, ...opts })
      : new TextRun({ text: part, ...opts }));
}
const P = (text, opts = {}) => (markEnd(), new Paragraph({ children: runs(text), spacing: { after: 120 }, ...opts }));
const H1 = (t, newPage = false) => (markEnd(), new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: newPage, children: [new TextRun(t)] }));
const H2 = (t) => (markEnd(), new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] }));
const H3 = (t) => (markEnd(), new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] }));
const bullet = (text, level = 0) =>
  new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60 } });
// each run of numbered items restarts at 1 (a new instance starts after any non-numbered paragraph)
let listInstance = 0, lastWasNumbered = false;
const markEnd = () => { lastWasNumbered = false; };
const numbered = (text) => {
  if (!lastWasNumbered) listInstance += 1;
  lastWasNumbered = true;
  return new Paragraph({ numbering: { reference: "numbers", level: 0, instance: listInstance }, children: runs(text), spacing: { after: 60 } });
};

let figNo = 0;
function figure(file, caption, source, widthPx = 600) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  figNo += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [
        new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18 }),
        new TextRun({ text: caption, size: 18 }),
        new TextRun({ text: `  Source: ${source}`, size: 16, italics: true, color: "666666" }),
      ],
    }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, caption) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, head, shade, last) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: ACCENT, type: ShadingType.CLEAR, color: "auto" }
      : shade ? { fill: "EAF1F8", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ keepNext: !last, children: runs(String(text), head ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }) })],
  });
  const out = [];
  if (caption) out.push(new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 },
    children: [new TextRun({ text: caption, bold: true, size: 19 })] }));
  out.push(new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
      ...rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, r.highlight, k === rows.length - 1)) })),
    ],
  }));
  out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}
const hl = (arr) => Object.assign(arr, { highlight: true });

// ---------- content ----------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 3000 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Project A: Café", size: 56, bold: true, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Business Analytics Report", size: 32, color: "404040" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Exploratory analysis, market basket analysis and Red Bull promotion effects", size: 24, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Group 1", size: 28, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[Group member names]", size: 22, color: "808080" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "November 2026", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, color: ACCENT })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// 1 Introduction
children.push(
  H1("1. Introduction"),
  P("We analysed one year of till data (1 April 2010 – 31 March 2011) from a café that belongs to a larger chain. The aim is to understand how the café performs, find patterns that can improve sales, and turn them into recommendations for management. The report follows the three parts of the assignment:"),
  bullet("**Part 1 – Exploratory Data Analysis:** data quality, sales trends, product and category performance, anomalies and feature engineering."),
  bullet("**Part 2 – Market Basket Analysis:** which products are bought together, and which pairs would make good combo deals."),
  bullet("**Part 3 – Additional exploration:** do Red Bull promotions really raise daily Red Bull revenue once store traffic, weekends and the menu price increase are taken into account?"),
  P("All analysis was done in Python (pandas, matplotlib, seaborn, statsmodels) in the notebook main.ipynb, with the calculations in separate modules in the scripts/ folder. **Every figure in this report is produced by that code**; the caption of each figure names the notebook step that creates it. Claude Code was used as an assistant in this project."),
);

// 2 Part 1
children.push(H1("2. Part 1 – Exploratory Data Analysis"));

children.push(
  H2("2.1 The data and how we cleaned it"),
  P("The raw file has **145,830 rows and 10 columns**: date, bill number, item, time, quantity, unit price (Rate), tax, discount, total and category. Each row is one product line on a bill. There are 69,982 bills, 580 product names and 9 categories, and no missing values. However, the money columns were stored as text with a comma as decimal separator, and several names and categories were inconsistent. We fixed these problems step by step:"),
  ...table(["Problem", "What we did", "Effect"], [
    ["Money stored as text (\"11,88\"), trailing spaces in column and category names", "Converted to numbers and dates; stripped spaces", "\"LIQUOR \" and \"LIQUOR\" no longer counted as two categories"],
    ["Broken or duplicate product names (junk characters, double spaces, typos)", "Compared names by similarity, price and dates in use; merged true duplicates only", "580 → 577 products"],
    ["Products in the wrong or several categories (e.g. 14 milkshakes under FOOD)", "Used each product's majority category; moved shakes to BEVERAGE, wines to WINES", "16,170 units moved from FOOD to BEVERAGE"],
    ["680 exact duplicate rows (same bill, item and second)", "Treated as the waiter entering an item several times: merged and summed the quantity", "145,830 → 145,052 rows (only 0.46% of revenue)"],
  ], [2900, 3500, 2626], "Table 1. Cleaning steps"),
  P("The cleaned data was saved at two levels: **cafe_clean_lines.csv** (one row per product per bill, 145,052 rows) and **cafe_clean_orders.csv** (one row per bill, 69,982 rows). The line level answers what was bought; the order level answers how big and what kind of orders are."),
);

children.push(
  H2("2.2 Headline figures"),
  ...table(["Measure", "Value", "Measure", "Value"], [
    ["Total revenue (incl. tax)", "32,805,895", "Number of orders (bills)", "69,982"],
    ["Average revenue per day", "89,879", "Average orders per day", "191.7"],
    ["Median order value", "391", "Median items per order", "2"],
    ["Items sold", "163,519", "Unique products", "577"],
  ], [2700, 1813, 2700, 1813], "Table 2. Headline figures for the year (currency units as recorded by the till)"),
  P("All key variables are **right-skewed**: most orders are small, with a long tail of large group bills (5% of orders are high outliers). We therefore use the median (391) rather than the mean (469) as the typical order value. The large bills are real sales (party bookings, group orders), so we kept them."),
);

children.push(
  H2("2.3 Sales over time"),
  H3("Trend and where the growth comes from"),
  P("An average day in Jan–Mar 2011 brought in about 97k, against 79k in Apr–Jun 2010: **+24%**. Splitting this growth shows that **only about a third comes from more orders** (+8% per day). The rest comes from larger orders (+5% items per order) and **higher prices** (+9% per item). Comparing the same 265 products in both quarters, prices went up by about 11%, mostly with the new menu of 7 January 2011. Orders per day peaked in December and by March 2011 were below April 2010: revenue is held up by prices, while visits are no longer growing."),
  ...figure("cell076_1.png", "Revenue growth split into orders per day, items per order and price per item, per quarter.", "main.ipynb, Step 5.1 (ts.plot_growth_drivers)"),
  H3("Daily and weekly cycle"),
  P("This is an **evening and night business**. Only 16% of orders come before 15:00, while **54% come after 19:00**; the busiest hour is 19:00 and it stays busy until midnight. On a weekly basis, Friday and Saturday nights are the strongest (about 99k per business day, where sales after midnight count for the previous evening), against 81–86k from Monday to Thursday."),
  ...figure("cell085_1.png", "Average orders per hour, units per category per hour, and hot versus cold drinks per hour.", "main.ipynb, Step 5.3.1 (ts.plot_hourly)"),
  ...figure("cell088_1.png", "Average revenue per weekday and orders per weekday and hour.", "main.ipynb, Step 5.3.2 (ts.plot_weekly)"),
  H3("Seasonality"),
  P("Revenue is lowest in May–June and peaks in December (New Year's Eve alone made about 250k, almost three times a normal day). The seasonal swing is driven by **hot drinks**, which more than double from May to December, while cold drinks stay flat all year. A decomposition of daily revenue attributes about 39% of the variation to the trend, 24% to the weekly cycle and 34% to noise."),
  ...figure("cell093_1.png", "Hot and cold drinks sold per month (left) and the share of cold drinks (right).", "main.ipynb, Step 5.3.3 (ts.plot_drinks_per_month)"),
);

children.push(
  H2("2.4 Product and category performance"),
  P("**Drinks bring in the volume, hookah brings in the money.** Beverages are 41% of units but only 25% of revenue, while tobacco (hookah) makes **44.5% of revenue** with 23% of the units, because a hookah earns about 389 per unit, three times a drink. Beverage, food and tobacco together make 92% of both units and revenue."),
  ...figure("cell057_1.png", "Units sold and revenue per category.", "main.ipynb, Step 4.5 (ds.plot_category_sales)"),
  P("**Nirvana Hookah Single** is the number one product both in units and revenue. Cappuccino is second in units but too cheap to reach the revenue top 5."),
  ...figure("cell054_1.png", "Top 10 products by units sold and by revenue.", "main.ipynb, Step 4.4 (ds.plot_top_products)"),
  P("Revenue is very concentrated: **17 products (3% of the menu) make half of the revenue** and 66 products make 80%. Many products with low yearly sales turned out to be one-day specials, new products or products retired with the new menu. Of the 313 products still on the 2011 menu, 78 sell less than once a week and together make only 0.9% of revenue. These are mostly merchandise, slow wines and 15 variants of speciality coffee that could be grouped."),
  ...figure("cell105_1.png", "Cumulative share of revenue by product (Pareto curve).", "main.ipynb, Step 6.2 (cat.plot_pareto)", 480),
  P("Hookah is the top earner in **every part of the day**, even at lunch (39–47% of revenue). Orders get larger through the day, from about 360 in the late morning to 520 at night, and alcohol grows from 1% to 9% of revenue at night."),
  ...figure("cell114_1.png", "Category mix of revenue per part of the day and per weekday.", "main.ipynb, Step 6.3 (cat.plot_category_mix)"),
);

children.push(
  H2("2.5 Relationships and anomalies"),
  bullet("**Price and units:** within liquor and wine, cheaper products sell more (Spearman ρ ≈ −0.75); for food, price has no significant effect; for hookah, **more expensive flavours sell more** (ρ = +0.51)."),
  bullet("**Categories move together:** daily sales of food, beverage and tobacco are correlated (0.54–0.66), because all are driven by the number of visitors."),
  ...table(["Anomaly", "Explanation", "Action"], [
    ["5 lines at a price of 0.01", "Free Mother's Day item on 9 May 2010", "Kept, no effect on revenue"],
    ["Biggest bill (14,231) on 25 Jan 2011", "Party booking for 23 people", "Kept, flagged as special day"],
    ["Two bill series: G and N", "N bills (9%) contain only alcohol, G bills never do: alcohol is billed separately", "Important for Part 2"],
    ["73 new and 125 retired products around January", "New menu from 7 January 2011", "Flagged as old/new menu"],
    ["New Year's Eve ~250k", "Holiday peak", "Flagged as special day"],
  ], [2800, 3926, 2300], "Table 3. Main anomalies found"),
);

children.push(
  H2("2.6 Feature engineering"),
  P("We turned these findings into new columns so that Parts 2 and 3 could use them directly:"),
  ...table(["Feature", "What it is", "Why"], [
    ["Daypart, Hour", "Late morning / Lunch / Afternoon / Evening / Night", "Ordering habits change through the day"],
    ["Business day, Weekday, Weekend", "Sales before 06:00 count for the previous evening", "Friday night stays Friday night"],
    ["Promo, Base product", "Deal type read from the name (1+1, 2+1, multi-pack, combo) and the product without the deal", "Promotions are not recorded in the Discount column"],
    ["Menu, Special day, Bill series", "Old/new menu, holidays and events, main or alcohol bill", "Anomalies from Step 6"],
    ["Order features", "Products, units, value and category flags per bill", "Bill-level analysis"],
  ], [2600, 3826, 2600], "Table 4. Engineered features"),
);

// 3 Part 2
children.push(
  H1("3. Part 2 – Market Basket Analysis", true),
  H2("3.1 Data choice and preparation"),
  P("Market basket analysis asks which products appear on the same bill. We used **cafe_clean_lines.csv**, because it keeps every product on every bill. The order-level file only keeps totals and category flags, so the individual products needed for combo deals are no longer visible there. Four transformations prepared the data:"),
  numbered("**Base product** instead of the raw name, so a product and its deal version count as the same product."),
  numbered("**Binary presence** per bill: a product counts once per bill, so one table ordering three of the same item does not inflate pair counts."),
  numbered("**Removing noise:** add-ons (ADD ON S) and merchandise, which are not bundled."),
  numbered("**Splitting by daypart**, because what people order changes through the day."),
  P("This leaves 144,226 lines on 69,663 bills. 45.5% of these bills contain only one product."),
  H2("3.2 Method: Apriori and the metrics"),
  P("We used the **Apriori** algorithm. It first keeps the products that appear on enough bills (minimum support), and then only builds pairs from those products, because a pair can never be more frequent than its least frequent product. We chose Apriori because each step is transparent and easy to verify. FP-Growth is faster on very large data but harder to follow, and ECLAT suits dense baskets, whereas our bills hold about two products on average."),
  ...table(["Metric", "Formula", "Meaning"], [
    ["Support (A, B)", "Bills with A and B ÷ all bills", "How common the pair is"],
    ["Confidence (A → B)", "Bills with A and B ÷ bills with A", "Of the bills with A, the share that also has B"],
    ["Lift (A, B)", "Support(A, B) ÷ (Support(A) × Support(B))", "How many times more often than chance (> 1 = real affinity)"],
  ], [2200, 3426, 3400], "Table 5. Association rule metrics"),
  P("**Thresholds.** Because almost half the bills hold one product, pair support is naturally low: a minimum support of 1% returns no pairs at all. We used **0.3% (at least 25 bills) for the whole year** and **0.4% (at least 10 bills) per daypart**, with a **minimum lift of 1.15**. We rank pairs by lift, not by volume: two very popular products often appear together just by chance (for example Nirvana Hookah + Poutine, lift 1.08), while lift shows genuine affinity."),
  H2("3.3 Results"),
  P("For the whole year we found **20 rules**. The strongest pair is **Red Bull + Sambuca hookah**: 612 bills together (support 0.88%), 18.2% of Red Bull bills also include Sambuca, and the lift is **2.86**, almost three times more than chance. The top three pairs are all Red Bull with a hookah flavour. The B.M.T. panini pairs with many products and is the most \"shareable\" food item."),
  ...figure("mba_overall_rules.png", "Top 10 product pairs of the year by lift, with the number of bills on which they appear together.", "outputs/apriori_combideals_overall_en.csv, plotted by scripts/report_figures.py"),
  P("Per daypart we found **67 rules**. The best pair differs strongly through the day, which is why one all-day combo would not fit:"),
  ...figure("mba_daypart_rules.png", "Best and runner-up pair per part of the day.", "outputs/apriori_combideals_per_daypart_en.csv, plotted by scripts/report_figures.py"),
  H2("3.4 Recommended combo deals"),
  ...table(["Daypart", "Combo deal", "Bills", "Lift", "Idea"], [
    ["Late morning (10–12h)", "Scrambled eggs + mineral water (1L)", "19", "4.67", "Breakfast bundle*"],
    ["Lunch (12–15h)", "B.M.T. panini + fries", "39", "6.58", "Lunch meal upgrade"],
    ["Afternoon (15–19h)", "Red Bull + Sambuca hookah", "191", "2.85", "Lounge bundle"],
    ["Evening (19–23h)", "Red Bull + Sambuca hookah", "209", "2.98", "Peak evening hookah + energy"],
    ["Night (23–03h)", "Nirvana hookah + mineral water (500ml)", "104", "3.25", "Late-night hydration bundle"],
  ], [1900, 2826, 800, 800, 2700], "Table 6. Recommended combo deal per daypart"),
  P("* The highest lift in the late morning is water + hash brown omelette, but it rests on only 10 bills; scrambled eggs + water (19 bills) is the safer choice.", { spacing: { after: 120 } }),
  P("**Cross-selling:** suggest Red Bull to customers ordering a hookah from 15:00 onwards, and fries with paninis at lunch. **Alcohol is invisible here:** because alcohol is billed on separate N bills (87% of which contain one drink), pairs of alcohol with food or hookah cannot be found in this data, although alcohol makes about 6.5% of revenue."),
);

// 4 Part 3
children.push(
  H1("4. Part 3 – Do Red Bull promotions increase revenue?", true),
  H2("4.1 Question and approach"),
  P("Part 1 showed that Red Bull deals (RED BULL 2+1 until early January 2011, then 2/3/5 RED BULL multi-packs, plus a few combos) are the café's most used promotion. At the same time the price of a regular Red Bull went up with the new menu, and traffic is higher at weekends. A simple comparison of promo and non-promo days could therefore be driven by busy days or by the higher price. Our question:"),
  P("**Do Red Bull promotions increase daily Red Bull revenue once store traffic, weekend patterns and the menu price increase are accounted for?**"),
  P("From cafe_clean_lines.csv we kept every line whose name or base product contains \"Red Bull\" (3,610 lines, 4,123 units, 1,046,680 in revenue) and rolled the data up to one row per business day (n = 366):"),
  ...table(["Variable", "Role", "Definition"], [
    ["rb_rev", "Outcome (Y)", "Total daily Red Bull revenue"],
    ["promo_share", "Predictor (X1)", "Share of the day's Red Bull units sold on a promotion (0 to 1)"],
    ["post_hike", "Control (X2)", "1 from 30 Dec 2010 onwards (new Red Bull price), 0 before"],
    ["weekend_num", "Control (X3)", "1 on Friday, Saturday and Sunday business days"],
    ["total_bills", "Control (X4)", "Number of bills in the whole café that day (store traffic)"],
  ], [1900, 1700, 5426], "Table 7. Variables of the daily Red Bull model"),
  P("We use the **share** of promo units rather than a yes/no promo-day flag, because on most days (275 of 366) at least some Red Bull is sold on a deal. We first fitted a **simple linear regression** with the promo share only, then a **multiple linear regression** with the three controls. Both use HC1 robust standard errors, because busier days vary more."),
  H2("4.2 Key operational findings"),
  bullet("**Price shift:** a regular Red Bull Energy Drink cost 125 before 30 December 2010 and 150 from then on (new menu)."),
  bullet("**Year-end spike:** on 30 and 31 December daily Red Bull revenue reached about 38k and 34k, more than ten times a normal day. Most of it came from RED BULL SHEESHA, a hookah flavour at 450 sold over New Year's Eve, not from a promotion."),
  bullet("**Promotion mix:** promo lines are 42.5% of Red Bull units but **58.5% of Red Bull revenue** (multi-buy 40.8%, multi-pack 11.8%, combo 5.9%), because a deal line sells several cans at a higher line price."),
  bullet("**Raw comparison:** days without any promo sales average about 1.7k of Red Bull revenue, against 3.1k on days with up to half the units on promo and 3.3k when more than half is on promo."),
  H2("4.3 Simple linear regression"),
  P("With the promo share as the only predictor, moving from 0% to 100% promo share goes together with **+1,534 daily Red Bull revenue** (p = 0.015). However, the model explains only **2.5%** of the day-to-day variation (R² = 0.025): the promo share alone says little without other factors."),
  ...figure("cell183_1.png", "Daily Red Bull revenue against the daily promo share, with the simple regression line and the averages per promo-share bin. The two points at the top left are 30–31 December.", "main.ipynb, Part 3 Step 3 (rba.plot_simple_regression)"),
  H2("4.4 Multiple linear regression"),
  ...table(["Variable", "Coefficient", "HC1 p-value", "Interpretation (controls held fixed)"], [
    ["Intercept", "−4,801.59", "0.150", "Baseline anchor"],
    hl(["Promo share", "+1,744.12", "0.001", "+10 percentage points of promo share adds about +174 daily revenue"]),
    ["Price hike (from 30 Dec 2010)", "+1,219.87", "0.003", "Higher baseline after the price increase"],
    ["Weekend", "−114.57", "0.824", "No separate weekend effect once traffic is included"],
    ["Bills in the café", "+34.85", "0.072", "Each extra bill adds about 35 (significant only at 10%)"],
  ], [2500, 1300, 1200, 4026], "Table 8. Multiple regression of daily Red Bull revenue (n = 366 days, HC1 robust standard errors)"),
  P("The promotion effect stays positive and becomes **stronger** once the controls are added: **+1,744 from 0% to 100% promo share (p = 0.001)**. The price increase adds about 1,220 per day, while the weekend has no effect of its own once the number of bills is in the model. The adjusted R² rises from 0.022 to **0.170**."),
  ...figure("cell186_1.png", "Partial effect of the promo share: daily Red Bull revenue with the effect of the price hike, weekend and store traffic removed.", "main.ipynb, Part 3 Step 4 (rba.plot_multiple_regression)"),
  ...table(["Metric", "Simple regression (k = 1)", "Multiple regression (k = 4)"], [
    ["SST (total variation)", "2,999,710,577.62", "2,999,710,577.62"],
    ["SSR (explained)", "74,584,498.73", "538,105,328.36"],
    ["SSE (unexplained)", "2,925,126,078.89", "2,461,605,249.26"],
    ["R²", "0.0249", "0.1794"],
    hl(["Adjusted R²", "0.0222", "0.1703"]),
  ], [3026, 3000, 3000], "Table 9. Variance decomposition and model fit"),
  H2("4.5 Sensitivity check"),
  P("The two New Year's Eve days are extreme values that have nothing to do with promotions. Without them, the promo-share effect rises to **+2,423 (p < 0.001)** and R² to 0.30. The conclusion does not depend on the year-end spike; if anything, those days hide part of the promo effect."),
  H2("4.6 Business implications"),
  bullet("Red Bull promotions are associated with higher daily Red Bull revenue even after controlling for price, weekends and traffic, so **continuing the multi-buy and multi-pack Red Bull deals** is supported by the data."),
  bullet("Deals fit best at **hookah tables**, where Part 2 found the strongest Red Bull pairs (Red Bull + Sambuca, lift 2.86)."),
  bullet("Before changing deal prices, management should check the **margin per deal**, which this data cannot show."),
);

// 5 Conclusions
children.push(
  H1("5. Conclusions and recommendations", true),
  numbered("**Hookah is the core of the business.** It makes 44.5% of revenue and leads in every part of the day; premium flavours sell well. Keep the hookah offer central and premium."),
  numbered("**Growth comes from prices, not visits.** Revenue rose 24%, but orders per day fell back after December. Monitor visit numbers after the price increases, and use the quiet late morning and lunch hours (16% of orders) as the main room for growth."),
  numbered("**Launch daypart-specific combos.** Panini + fries at lunch, Red Bull + Sambuca hookah from 15:00, and hookah + water at night are the strongest pairs."),
  numbered("**Keep the multi-buy and multi-pack Red Bull deals, aimed at hookah tables.** Each extra 10 percentage points of promo share goes together with about +174 daily Red Bull revenue, after controlling for the price increase, weekends and traffic."),
  numbered("**Trim the menu lightly.** 17 products make half of the revenue, but the truly slow products on the current menu are few: group the speciality coffees and review slow wines and merchandise."),
  numbered("**Staff for Friday and Saturday nights**, including the hours after midnight, and plan for December and New Year's Eve."),
);

// 6 Limitations
children.push(
  H1("6. Limitations"),
  bullet("**No customer ID or party size:** a bill is not a customer, so we cannot study repeat visits, and a high lift can reflect several people at one table."),
  bullet("**No cost data:** all conclusions are about revenue, not profit; bundle discounts must be checked against margins."),
  bullet("**Alcohol is billed separately**, so alcohol cannot be linked to food or hookah in the same visit."),
  bullet("**Association is not causation:** neither the basket rules nor the promotion regression are controlled experiments. In Part 3, days with more promo sales may also be days when customers wanted more Red Bull anyway, and 83% of the daily variation remains unexplained (adjusted R² = 0.17)."),
  bullet("**Ticket size vs. volume:** a deal line carries a higher price than a single can, so part of the extra Red Bull revenue is a bigger ticket per order rather than more cans sold."),
  bullet("**One café, one year (2010–2011):** we cannot separate real growth from a yearly cycle, and menu and prices must be checked before acting."),
  bullet("**Cleaning choices:** some fixes (merging duplicate rows, moving milkshakes to BEVERAGE) are judgement calls. Each fix is documented in the notebook so it can be checked or undone."),
);

// ---------- document ----------
const heading = (size, color, before, after) => ({
  run: { font: FONT, size, bold: true, color },
  paragraph: { spacing: { before, after }, keepNext: true },
});
const doc = new Document({
  creator: "Group 1",
  title: "Project A: Café – Business Analytics Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...heading(32, ACCENT, 360, 160), paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...heading(26, ACCENT, 240, 120), paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...heading(23, "404040", 180, 80), paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Group 1 – Project A: Café    |    page ", size: 16, color: "808080" }),
                 new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" })] })] }) },
    children,
  }],
});

const out = process.env.REPORT_OUT || path.join(__dirname, "Group 1 - Project A Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("Wrote", out); });
