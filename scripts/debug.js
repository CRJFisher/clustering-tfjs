const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { SpectralClustering } = require('../src');

const fixturePath = process.argv[2];
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

async function run() {
  const ctrParams = { ...fixture.params };
  const model = new SpectralClustering(ctrParams);
  const labels = await model.fitPredict(fixture.X);
  console.log('our labels first 30', labels.slice(0,30));
  console.log('labels unique count', new Set(labels).size);
  console.log('fixture unique', Array.from(new Set(fixture.labels)).length);
  // compute ARI
  function comb2(x){return x*(x-1)/2;}
  function ari(a,b){
    const n=a.length; const mapA=new Map(), mapB=new Map(); let nextA=0,nextB=0; const cont=[];
    for(let i=0;i<n;i++){
      const la=a[i], lb=b[i];
      if(!mapA.has(la)){mapA.set(la,nextA++); cont.push(Array(nextB).fill(0));}
      const idxA=mapA.get(la);
      if(!mapB.has(lb)){
        mapB.set(lb,nextB++); cont.forEach(r=>r.push(0));
      }
      const idxB=mapB.get(lb);
      cont[idxA][idxB]++;
    }
    const ai=cont.map(r=>r.reduce((s,v)=>s+v,0));
    const bj=cont[0].map((_,j)=>cont.reduce((s,row)=>s+row[j],0));
    let sumComb=0; for(const row of cont){for(const v of row) sumComb+=comb2(v);}
    const sumAi=ai.reduce((s,v)=>s+comb2(v),0);
    const sumBj=bj.reduce((s,v)=>s+comb2(v),0);
    const expected=(sumAi*sumBj)/comb2(n);
    const max=(sumAi+sumBj)/2;
    if(max===expected) return 0;
    return (sumComb - expected)/(max - expected);
  }
  console.log('ARI', ari(labels, fixture.labels));
}

run();
