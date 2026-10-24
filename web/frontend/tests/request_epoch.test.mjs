import assert from 'node:assert/strict';
import { createRequestEpoch } from '../src/requestEpoch.js';

const epoch = createRequestEpoch();
const requestA = epoch.begin();
const requestB = epoch.begin();

assert.equal(epoch.isCurrent(requestA), false, 'stale request A must be rejected');
assert.equal(epoch.isCurrent(requestB), true, 'latest request B must be accepted');

epoch.invalidate();
assert.equal(epoch.isCurrent(requestB), false, 'selection change invalidates pending request');

const requestC = epoch.begin();
assert.equal(epoch.isCurrent(requestC), true, 'new request after selection is current');
console.log('request epoch regression PASS: stale result/error/progress cannot commit');
