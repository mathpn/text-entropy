import { AutoTokenizer, AutoModelForCausalLM } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';
import { softmax } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/src/utils/maths.js';

let model = null;
let tokenizer = null;

async function initializeModel() {
  try {
    tokenizer = await AutoTokenizer.from_pretrained(
      "Xenova/gpt2"
    );
    model = await AutoModelForCausalLM.from_pretrained(
      "Xenova/gpt2", { model_file_name: "decoder_model", dtype: "q8" },
    );
    self.postMessage({ type: 'modelReady' });
  } catch (error) {
    self.postMessage({ type: 'error', error: error.message });
  }
}

async function processText(text, contextWindow, batchSize, maxTokens) {

  const batchStrings = [];
  const batchTargets = [];
  try {
    // Add bos token to begginning of text
    const bosToken = tokenizer.decode([tokenizer.bos_token_id]);
    const tokens = tokenizer(bosToken + text, { add_special_tokens: false });
    const inputIds = tokens.input_ids;
    const numTokens = Math.min(inputIds.data.length, maxTokens);

    let logProbSum = 0;
    let count = 0;

    self.postMessage({ type: 'progress', progress: 0 });

    for (let i = 1; i < numTokens; i++) {
      const startIdx = Math.max(0, i - contextWindow);
      const batchString = tokenizer.decode(inputIds.slice(null, [startIdx, i]));
      const target = inputIds.slice(null, i).item();
      batchStrings.push(batchString);
      batchTargets.push(target);
    }

    // Create batches with batchSize elements from batchString
    for (let i = 0; i < batchStrings.length; i += batchSize) {
      const endIdx = Math.min(i + batchSize, batchStrings.length);
      const batch = batchStrings.slice(i, endIdx);
      const batchTokens = tokenizer(batch, { padding: true });
      const batchInput = {
        input_ids: batchTokens.input_ids,
        attention_mask: batchTokens.attention_mask,
        targets: batchTargets.slice(i, endIdx)
      }

      const results = await processBatch(batchInput, i * batchSize, (i + 1) * batchSize, i);
      self.postMessage({ type: 'results', results });

      // Update progress
      const progress = (i / batchStrings.length) * 100;
      self.postMessage({ type: 'progress', progress });

      for (const result of results) {
        logProbSum += result.logProb;
        count++;
      }

      // Send perplexity
      if (count > 0) {
        const perp = Math.exp(-logProbSum / count);
        self.postMessage({
          type: 'perplexity',
          perplexity: perp
        });
      }
    }

    self.postMessage({ type: 'progress', progress: 100 });
  } catch (error) {
    self.postMessage({ type: 'error', error: error.message });
  }


}

async function processBatch(inputs, startIdx) {
  const outputs = await model(inputs);
  const logits = outputs.logits;
  const attention_mask = inputs.attention_mask;

  const batchSize = logits.dims[0];
  const results = [];

  for (let i = 0; i < batchSize; ++i) {
    const seqMaskSlice = attention_mask.slice(i, null);
    const seqMaskData = Array.from(seqMaskSlice.data);
    const seqLength = seqMaskData.reduce((acc, val) => acc + val, BigInt(0));
    const lastTokenIdx = seqLength - BigInt(1);

    if (lastTokenIdx !== -1) {
      const lastTokenLogits = logits.slice(i, Number(lastTokenIdx), null);
      const probs = softmax(lastTokenLogits.data);
      const target = inputs.targets[i];
      const targetProb = probs[target];

      results.push({
        index: i + startIdx,
        probability: Math.round(targetProb * 10000) / 100,
        logProb: Math.round(Math.log(targetProb) * 1000) / 1000,
        tokenString: tokenizer.decode([target])
      });
    }
  }
  return results;
}

self.onmessage = async function (e) {
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