import { AutoTokenizer, AutoModelForCausalLM } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';

let model = null;
let tokenizer = null;

async function initializeModel() {
  try {
    tokenizer = await AutoTokenizer.from_pretrained(
      "HuggingFaceTB/SmolLM2-135M-Instruct"
    );
    model = await AutoModelForCausalLM.from_pretrained(
      "HuggingFaceTB/SmolLM2-135M-Instruct"
    );
    self.postMessage({ type: 'modelReady' });
  } catch (error) {
    self.postMessage({ type: 'error', error: error.message });
  }
}

async function processText(text, contextWindow, batchSize, maxTokens) {
  // Tokenize the text
  const tokens = await tokenizer(text);
  const inputIds = tokens.input_ids.data;
  const numTokens = Math.min(inputIds.length, maxTokens);

  // Process tokens in batches
  for (let startIdx = 0; startIdx < numTokens; startIdx += batchSize) {
    const endIdx = Math.min(startIdx + batchSize, numTokens);
    const batchInputs = [];

    // Create batch input tensors
    for (let i = startIdx; i < endIdx; i++) {
      const contextStart = Math.max(0, i - contextWindow);
      batchInputs.push({
        input_ids: tokens.input_ids.slice([0, 1], [contextStart, i]),
        attention_mask: tokens.attention_mask.slice([0, 1], [contextStart, i]),
        target: inputIds[i]
      });
    }

    // Process the batch
    const results = await processBatch(batchInputs, startIdx);
    self.postMessage({ type: 'results', results });

    // Add a small delay between batches
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}

async function processBatch(batchInputs, startIdx) {
  const results = [];
  for (let i = 0; i < batchInputs.length; i++) {
    const input = batchInputs[i];
    const target = input.target;
    
    const outputs = await model(input);
    const logits = outputs.logits;
    const vocabSize = model.config.vocab_size;
    const lastTokenLogits = logits.data.slice(-vocabSize);
    
    const maxLogit = Math.max(...lastTokenLogits);
    const expLogits = lastTokenLogits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    
    const targetProb = expLogits[target] / sumExp || 1e-10;
    const perplexity = Math.min(1 / targetProb, 1000);
    
    results.push({
      index: startIdx + i,
      perplexity,
      tokenString: tokenizer.decode([target])
    });
  }
  return results;
}

self.onmessage = async function(e) {
  const { type, data } = e.data;
  
  switch (type) {
    case 'init':
      await initializeModel();
      break;
    case 'process':
      await processText(
        data.text,
        data.contextWindow,
        data.batchSize,
        data.maxTokens
      );
      break;
  }
}; 