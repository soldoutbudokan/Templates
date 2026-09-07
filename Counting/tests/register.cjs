// Use the project's existing TypeScript compiler; no test dependency is needed.
const fs = require('node:fs');
const ts = require('typescript');
require.extensions['.ts'] = function register(module, filename) {
  const source = fs.readFileSync(filename, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 }
  });
  module._compile(outputText, filename);
};
