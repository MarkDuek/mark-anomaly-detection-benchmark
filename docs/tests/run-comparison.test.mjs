import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../src/components/runs/RunComparison.astro", import.meta.url), "utf8");
const between = (start, end) => source.slice(source.indexOf(start), source.indexOf(end));
const window = {
  location: new URL("https://example.com/runs/?other=keep#compare"),
  history: { replaceState(_state, _title, url) { window.location = new URL(url); } },
};
let inputs = [];
const context = vm.createContext({
  URL, URLSearchParams, window,
  runs: Object.fromEntries(["a1", "a2", "a3", "b1", "b2", "b3"].map((id) => [id, { dataset_id: id[0] }])),
  datasets: ["a", "b"],
  datasetSelector: { value: "a" },
  runSelection: {},
  render() {}, renderCurves() {},
  options: {
    replaceChildren(...labels) { inputs = labels.map((label) => label.querySelector()); },
    querySelectorAll() { return inputs.filter((input) => input.checked); },
  },
  document: {
    createElement() {
      return {
        querySelector() {
          return this.input ??= {
            value: this.innerHTML.match(/value="([^"]+)"/)[1],
            checked: this.innerHTML.includes(" checked>"),
            addEventListener() {},
          };
        },
      };
    },
  },
});
vm.runInContext(between("  const selected =", "  const entries ="), context);
vm.runInContext(between("  const renderOptions =", "  selectAllRuns.addEventListener"), context);
const reload = () => {
  context.datasetSelector.value = "a";
  vm.runInContext(`{${between("    const requestedDataset =", "    datasetSelector.addEventListener")}}`, context);
  vm.runInContext("renderOptions()", context);
};
const selected = () => inputs.filter((input) => input.checked).map((input) => input.value);

reload();
assert.deepEqual(selected(), ["a1", "a2"]);
context.datasetSelector.value = "b";
vm.runInContext("renderOptions()", context);
inputs.forEach((input) => input.checked = input.value === "b3");
vm.runInContext("refreshSelection()", context);
reload();
assert.equal(context.datasetSelector.value, "b");
assert.deepEqual(selected(), ["b3"]);
assert.equal(window.location.searchParams.get("other"), "keep");
assert.equal(window.location.hash, "#compare");

inputs.forEach((input) => input.checked = false);
vm.runInContext("refreshSelection()", context);
reload();
assert.deepEqual(selected(), []);

window.location = new URL("https://example.com/runs/?dataset=b&run=missing&run=a1&run=b2");
reload();
assert.deepEqual(selected(), ["b2"]);
assert.deepEqual(window.location.searchParams.getAll("run"), ["b2"]);
window.location = new URL("https://example.com/runs/?dataset=missing&run=b3");
reload();
assert.equal(context.datasetSelector.value, "a");
assert.deepEqual(selected(), ["a1", "a2"]);
console.log("Run comparison selection reload checks passed.");
