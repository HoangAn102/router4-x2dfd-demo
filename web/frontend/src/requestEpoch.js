// Guards asynchronous UI results against a newer selection/request.
export function createRequestEpoch() {
  let current = 0;

  return {
    begin() {
      current += 1;
      return current;
    },
    invalidate() {
      current += 1;
      return current;
    },
    isCurrent(token) {
      return token === current;
    },
  };
}
