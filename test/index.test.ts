import { hello } from "../src/index";

describe("hello", () => {
  it("greets the world", () => {
    expect(hello()).toBe("Hello, world!");
  });
});

