/* BINAC v9.6 · 浏览器 JS 引擎
 * 架构: Transformer (Emb + Pos + N×Block + LNf + head)
 * 兼容: window.BINAC_BRAIN = {...v9.6_memory.json...}
 * 用法: <script src="binac_v9.6_memory.js"></script>
 *       <script src="binac_v96.js"></script>
 *       BINAC.ask("hello")   -> {answer, source, sim, matched_q}
 *
 * 与 Python binac_v9_chat.py 前向一致（已逐位校验）
 */
(function (root) {
  var PAD = 0, EOS = 1, UNK = 2;

  function softmax(v) {
    var mx = -Infinity, i;
    for (i = 0; i < v.length; i++) if (v[i] > mx) mx = v[i];
    var s = 0, o = new Float64Array(v.length);
    for (i = 0; i < v.length; i++) { o[i] = Math.exp(v[i] - mx); s += o[i]; }
    for (i = 0; i < v.length; i++) o[i] /= s;
    return o;
  }

  /* ---------- 矩阵工具 ---------- */
  // x: (n,d)  W: (d,out)  b: (out)  -> (n,out)
  function matmul(x, W, b, n, d, out) {
    var o = new Float64Array(n * out), i, j, k, sum;
    for (i = 0; i < n; i++) {
      for (j = 0; j < out; j++) {
        sum = b ? b[j] : 0;
        for (k = 0; k < d; k++) sum += x[i * d + k] * W[k][j];
        o[i * out + j] = sum;
      }
    }
    return o;
  }

  // (n,d) 按最后一个维度做 LayerNorm
  function layernorm(x, g, b, n, d, eps) {
    var o = new Float64Array(n * d), i, j, mu, vr, s;
    for (i = 0; i < n; i++) {
      mu = 0;
      for (j = 0; j < d; j++) mu += x[i * d + j];
      mu /= d;
      vr = 0;
      for (j = 0; j < d; j++) { s = x[i * d + j] - mu; vr += s * s; }
      vr /= d;
      s = 1 / Math.sqrt(vr + eps);
      for (j = 0; j < d; j++) o[i * d + j] = (x[i * d + j] - mu) * s * g[j] + b[j];
    }
    return o;
  }

  /* ---------- 分词 ---------- */
  function makeTok(b) {
    var w2i = b.word2id, s2i = b.sub2id, sub = b.sub, total = b.total;
    var rev = {};
    for (var k in w2i) rev[w2i[k]] = k;
    for (var k2 in s2i) rev[s2i[k2]] = k2;
    return {
      total: total, w2i: w2i, s2i: s2i, sub: sub, rev: rev,
      tok: function (txt) {
        var ws = (txt.toLowerCase().match(/[a-z]+/g) || []), o = [], n, w, i, m, g;
        for (n = 0; n < ws.length; n++) {
          w = ws[n];
          if (w2i[w] !== undefined) { o.push(w2i[w]); continue; }
          i = 0;
          while (i < w.length) {
            g = false;
            for (m = 0; m < sub.length; m++) {
              if (w.substr(i, sub[m].length) === sub[m]) {
                o.push(s2i[sub[m]]); i += sub[m].length; g = true; break;
              }
            }
            if (!g) { o.push(w2i[w[i]] !== undefined ? w2i[w[i]] : UNK); i++; }
          }
        }
        return o;
      },
      merge: function (ids) {
        var out = [], buf = '', i, w;
        for (i = 0; i < ids.length; i++) {
          if (ids[i] === PAD || ids[i] === EOS) break;
          w = this.rev[ids[i]] || '?';
          if (/^[a-z]$/.test(w)) { buf += w; }
          else { if (buf) { out.push(buf); buf = ''; } out.push(w); }
        }
        if (buf) out.push(buf);
        return out.join(' ').trim();
      }
    };
  }

  /* ---------- IDF 加权检索（与 Python Retriever 一致） ---------- */
  function makeRetriever(qa) {
    if (!qa || !qa.length) return null;
    var N = qa.length;

    function feats(t) {
      var w = (t.toLowerCase().match(/[a-z]+/g) || []), s = {}, i;
      for (i = 0; i < w.length; i++) {
        s['u:' + w[i]] = 1;
        if (i + 1 < w.length) s['b:' + w[i] + ' ' + w[i + 1]] = 1;
      }
      return Object.keys(s);
    }

    var F = qa.map(function (p) { return feats(p[0]); });
    var df = {};
    F.forEach(function (f) { f.forEach(function (w) { df[w] = (df[w] || 0) + 1; }); });
    var idf = {}, defIdf = Math.log(N / 1) + 0.05;
    for (var k in df) idf[k] = Math.log(N / (1 + df[k])) + 0.05;
    function W(x) { return idf[x] !== undefined ? idf[x] : defIdf; }

    // 与 Python Retriever.sim 完全一致:
    //   num = Σ idf(交集)   den = Σ idf(并集)
    function sim(a, b) {
      var bs = {}, seen = {}, i, num = 0, den = 0;
      for (i = 0; i < b.length; i++) bs[b[i]] = 1;
      for (i = 0; i < a.length; i++) seen[a[i]] = 1;
      for (i = 0; i < b.length; i++) seen[b[i]] = 1;
      for (var k in seen) den += W(k);
      for (i = 0; i < a.length; i++) if (bs[a[i]]) num += W(a[i]);
      return den > 0 ? num / den : 0;
    }

    return {
      thresh: 0.40,
      answer: function (q) {
        var fq = feats(q), best = 0, bi = -1, i, s;
        for (i = 0; i < F.length; i++) {
          if (!F[i].length || !fq.length) continue;
          s = sim(fq, F[i]);
          if (s > best) { best = s; bi = i; }
        }
        if (bi >= 0 && best >= this.thresh) return [qa[bi][1], best, qa[bi][0]];
        return [null, best, null];
      }
    };
  }

  /* ---------- 主引擎 ---------- */
  var B = null, NET = null, TOK = null, RET = null, CFG = null;

  function load(brain) {
    B = brain;
    NET = brain.net;
    CFG = { D: brain.d, H: brain.h, NL: brain.nl, CTX: brain.context, DFF: brain.dff };
    TOK = makeTok(brain);
    var qa = brain.qa ? brain.qa.slice() : [];
    var have = {};
    for (var i = 0; i < qa.length; i++) have[qa[i][0].toLowerCase()] = 1;
    for (var k = 0; k < SELF_QA.length; k++) {
      if (!have[SELF_QA[k][0]]) { qa.push([SELF_QA[k][0], SELF_QA[k][1]]); }
    }
    RET = makeRetriever(qa);
    return api;
  }

  // 前向：ids(长度C) -> 最后位置 vocab 概率
  function forwardLast(ids) {
    var C = ids.length, D = CFG.D, H = CFG.H, hd = D / H, i, j, k, h;
    var x = new Float64Array(C * D);
    for (i = 0; i < C; i++) {
      for (j = 0; j < D; j++) x[i * D + j] = NET.Emb[ids[i]][j] + NET.Pos[i][j];
    }
    var qa, ka, va, att, ctxv, tmp;
    for (var L = 0; L < CFG.NL; L++) {
      var p = 'L' + L + '_';
      // LN1
      var h1 = layernorm(x, NET[p + 'ln1g'], NET[p + 'ln1b'], C, D, 1e-5);
      // QKV
      qa = matmul(h1, NET[p + 'Wq'], null, C, D, D);
      ka = matmul(h1, NET[p + 'Wk'], null, C, D, D);
      va = matmul(h1, NET[p + 'Wv'], null, C, D, D);
      // 多头注意力（causal mask）
      ctxv = new Float64Array(C * D);
      for (h = 0; h < H; h++) {
        for (i = 0; i < C; i++) {
          // 第 i 个位置，只能看 j<=i
          var sc = new Float64Array(i + 1), s;
          for (j = 0; j <= i; j++) {
            s = 0;
            for (k = 0; k < hd; k++) s += qa[i * D + h * hd + k] * ka[j * D + h * hd + k];
            sc[j] = s / Math.sqrt(hd);
          }
          var pr = softmax(sc);
          for (k = 0; k < hd; k++) {
            s = 0;
            for (j = 0; j <= i; j++) s += pr[j] * va[j * D + h * hd + k];
            ctxv[i * D + h * hd + k] = s;
          }
        }
      }
      // + O proj
      var o1 = matmul(ctxv, NET[p + 'Wo'], null, C, D, D);
      for (i = 0; i < C * D; i++) x[i] += o1[i];
      // LN2 + FFN
      var h2 = layernorm(x, NET[p + 'ln2g'], NET[p + 'ln2b'], C, D, 1e-5);
      var ff1 = matmul(h2, NET[p + 'W1'], NET[p + 'b1'], C, D, CFG.DFF);
      for (i = 0; i < ff1.length; i++) if (ff1[i] < 0) ff1[i] = 0;
      var ff2 = matmul(ff1, NET[p + 'W2'], NET[p + 'b2'], C, CFG.DFF, D);
      for (i = 0; i < C * D; i++) x[i] += ff2[i];
    }
    var xf = layernorm(x, NET.lnfg, NET.lnfb, C, D, 1e-5);
    var lg = matmul(xf, NET.Wout, NET.bout, C, D, B.total);
    // 取最后位置
    var out = new Float64Array(B.total);
    for (i = 0; i < B.total; i++) out[i] = lg[(C - 1) * B.total + i];
    return softmax(out);
  }

  var DROP = { hello: 1, hi: 1, hey: 1, good: 1, morning: 1, afternoon: 1,
    evening: 1, excuse: 1, sorry: 1, so: 1, and: 1, ok: 1, okay: 1, well: 1,
    now: 1, um: 1, actually: 1, just: 1, please: 1, binac: 1, really: 1 };

  var EXPAND = [['whats', 'what is'], ['hows', 'how is'], ['im', 'i am'],
    ['dont', 'do not'], ['cant', 'can not'], ['thats', 'that is'],
    ['its', 'it is'], ['youre', 'you are'], ['isnt', 'is not'],
    ['arent', 'are not'], ['wont', 'will not'], ['ive', 'i have'],
    ['lets', 'let us']];

  function stripCands(text) {
    var w = text.toLowerCase().split(/\s+/), cands = [text], i = 0, rest;
    while (i < w.length && DROP[w[i].replace(/[,]/g, '')] && w.length - i > 1) {
      i++;
      rest = w.slice(i).join(' ').trim();
      if (rest) cands.push(rest);
    }
    ['please', 'thanks', 'thank you', 'binac'].forEach(function (suf) {
      var low = text.toLowerCase();
      if (low.length > suf.length && low.substr(-suf.length - 1) === ' ' + suf) {
        var r = text.substr(0, text.length - suf.length - 1).trim();
        if (r) cands.push(r);
      }
    });
    var base = cands[cands.length - 1] || text;
    EXPAND.forEach(function (p) {
      var re = new RegExp('\\b' + p[0] + '\\b', 'g');
      if (re.test(base)) cands.push(base.replace(re, p[1]));
    });
    var seen = {}, u = [];
    for (i = 0; i < cands.length; i++) {
      if (cands[i] && !seen[cands[i]]) { seen[cands[i]] = 1; u.push(cands[i]); }
    }
    return u.length ? u : [text];
  }

  // 对原句和所有剥离候选都检索，取最高分
  function retrieveBest(text) {
    var best = [null, 0, null], cs = stripCands(text), i, r;
    for (i = 0; i < cs.length; i++) {
      r = RET.answer(cs[i]);
      if (r[0] !== null && r[1] > best[1]) { best = r; }
      if (best[1] >= 0.999) break;
    }
    return best;
  }

  // 中文避让：CJK 汉字 + 中文标点（刻意不含全角区 U+FF00-FFEF，
  // 否则日文假名 ｱｲｳ 会被误判成中文）
  var ZH_HAN = /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/;
  var ZH_PUNCT = /[\u3000-\u303f\uff01\uff1f\uff0c\uff1b\uff1a\uff08\uff09\uff02\u201c\u201d\u2018\u2019]/;
  var ZH_ASCII = /[a-z0-9]/i;
  // 分级判定：含汉字->中文；只有中文标点且无字母->中文；有字母一律按英文
  // （手机中文输入法打 Hello. 句号会变 。，不能因此拒绝英文）
  function isChinese(t) {
    if (ZH_HAN.test(t)) return true;
    if (ZH_PUNCT.test(t) && !ZH_ASCII.test(t)) return true;
    return false;
  }
  var ZH_SNARK = [
    "ERR 0xE4 charset mismatch this endpoint expects UTF-8 english tokens",
    "HTTP 415 unsupported media type my tokenizer only parses a to z try english",
    "parse error at index 0 expected latin alphabet received CJK codepoint",
    "frontend blocked charset utf8-cjk rejected by binary pipeline switch to english",
    "ERR charset rejected downstream model is english only request denied",
    "tokenizer cannot map those glyphs to my vocabulary please resubmit in english",
    "416 range not satisfiable my dictionary has no entries in that script",
    "input rejected by language filter english or binary only no exceptions"
  ];



  /* ---------- 数学计算器（算术走规则，不经过模型） ---------- */
  var MATH_PAT = /(-?\d+(?:\.\d+)?)\s*([-+*/x×÷])\s*(-?\d+(?:\.\d+)?)/;
  var WORD_OPS = [
    [/\btimes\b|\bmultiplied by\b/gi, '*'],
    [/\bplus\b|\badded to\b/gi, '+'],
    [/\bminus\b|\bsubtract\b/gi, '-'],
    [/\bdivided by\b|\bover\b/gi, '/']
  ];
  var OPS = {
    '+': function (a, b) { return a + b; },
    '-': function (a, b) { return a - b; },
    '*': function (a, b) { return a * b; },
    'x': function (a, b) { return a * b; },
    '×': function (a, b) { return a * b; },
    '/': function (a, b) { return b === 0 ? null : a / b; },
    '÷': function (a, b) { return b === 0 ? null : a / b; }
  };

  function tryMath(text) {
    var t = text, i;
    for (i = 0; i < WORD_OPS.length; i++) t = t.replace(WORD_OPS[i][0], WORD_OPS[i][1]);
    var m = MATH_PAT.exec(t);
    if (!m) return null;
    var a = parseFloat(m[1]), op = m[2], b = parseFloat(m[3]);
    var f = OPS[op.toLowerCase()];
    if (!f) return null;
    var r = f(a, b);
    if (r === null) return 'you cannot divide by zero';
    if (Math.abs(r - Math.round(r)) < 1e-9) { r = Math.round(r); a = Math.round(a); b = Math.round(b); }
    var sym = (op === 'x' || op === '×') ? '*' : (op === '÷' ? '/' : op);
    return a + ' ' + sym + ' ' + b + ' is ' + r;
  }

  /* ---------- 自指问答（模型必须知道自己是谁） ---------- */
  var SELF_QA = [
    ['what is binac', 'binac is a small binary brain that runs on numbers it answers simple questions and tells short stories'],
    ['what is binac', 'i am binac a tiny language model built from zeros and ones i am still learning so my answers can be rough'],
    ['who is binac', 'binac is the small brain you are talking to right now'],
    ['what are you', 'i am binac a small binary brain made of numbers and code'],
    ['are you a robot', 'yes i am a small program made of numbers not a person'],
    ['are you human', 'no i am not human i am a small program made of numbers'],
    ['are you ai', 'i am a very small language model not a big one but yes'],
    ['what model are you', 'i am binac a tiny transformer trained on a small set of questions'],
    ['how do you work', 'i turn your words into numbers then guess the next word one at a time until the answer feels finished'],
    ['what can you do', 'i can answer simple questions tell short stories do small sums and talk about words']
  ];

  /* ---------- 长句拆解 + 规律拼装 ---------- */
  // 意图：动作词 -> 模板类型
  var CINTENT = { sentence: 'sentence', sentences: 'sentence',
    describe: 'describe', description: 'describe',
    explain: 'explain', meaning: 'explain', mean: 'explain',
    story: 'story', poem: 'story', joke: 'story', write: 'story', sing: 'story',
    like: 'opinion', think: 'opinion', feel: 'opinion', love: 'opinion',
    hate: 'opinion', better: 'compare', worse: 'compare',
    how: 'howto', cook: 'howto', fix: 'howto', build: 'howto' };

  var CSTOP = ('a an the is are was were am be been being do does did doing '
    + 'make makes making tell tells telling about with for to of in on at by '
    + 'me you i my your it its this that these those and or but what how why '
    + 'who when where which can could would should will shall may might '
    + 'please thanks thank okay ok hey hello hi sentence sentences word words '
    + 'example examples some any give gives gave show shows use uses using').split(' ');
  var CVERBS = ('cook cooks cooking make makes making do does did doing '
    + 'describe describes describing explain explains tell tells telling '
    + 'give gives use uses using show shows find finds fix fixes build builds '
    + 'write writes read reads say says speak speaks talk talks walk walks '
    + 'run runs eat eats drink drinks play plays work works think thinks '
    + 'feel feels like likes love loves want wants need needs get gets '
    + 'take takes put puts open opens close closes start starts stop stops '
    + 'learn learns teach teaches help helps ask asks answer answers').split(' ');

  // 名词属性：a=形容词 v=动词 p=地点 isa=类别
  var CNOUNS = {
    apple: {a:'red sweet round crisp', v:'eat pick grow share bite', p:'tree basket kitchen garden', isa:'a fruit'},
    banana: {a:'yellow soft sweet long', v:'peel eat slice share', p:'tree bowl kitchen', isa:'a fruit'},
    water: {a:'clear cold clean deep', v:'drink pour flow boil', p:'glass river bottle sea', isa:'a liquid'},
    sea: {a:'deep blue wide wild', v:'swim sail cross watch', p:'shore coast horizon', isa:'a large body of water'},
    ocean: {a:'deep vast blue dark', v:'cross swim sail explore', p:'shore coast deep', isa:'a large body of water'},
    mountain: {a:'tall high rocky snowy', v:'climb hike cross see', p:'valley cloud trail', isa:'a high place of rock'},
    tree: {a:'tall green old leafy', v:'climb plant grow rest', p:'forest garden hill field', isa:'a plant'},
    flower: {a:'small bright colorful soft', v:'pick grow smell give', p:'garden field meadow vase', isa:'a plant'},
    sun: {a:'bright warm golden hot', v:'rise shine set warm', p:'sky morning horizon', isa:'a star'},
    moon: {a:'bright pale round quiet', v:'rise shine watch follow', p:'sky night window', isa:'a natural satellite'},
    sky: {a:'clear blue wide grey', v:'watch clear darken brighten', p:'above horizon cloud', isa:'the air above us'},
    rain: {a:'soft heavy cold gentle', v:'fall pour stop drip', p:'window roof street sky', isa:'water from the sky'},
    snow: {a:'white cold soft deep', v:'fall melt cover drift', p:'ground hill roof window', isa:'frozen water'},
    wind: {a:'cold strong gentle wild', v:'blow rise move whisper', p:'tree field sea hill', isa:'moving air'},
    fire: {a:'hot bright warm wild', v:'burn light warm spread', p:'stove forest hearth', isa:'a chemical reaction'},
    dog: {a:'friendly loyal quick small', v:'run bark play follow', p:'house park yard street', isa:'an animal'},
    cat: {a:'quiet soft small clever', v:'sleep jump hunt play', p:'house window chair garden', isa:'an animal'},
    bird: {a:'small quick free colorful', v:'fly sing hop build', p:'tree sky branch nest', isa:'an animal'},
    fish: {a:'quick silver small quiet', v:'swim catch cook watch', p:'river sea tank pond', isa:'an animal'},
    book: {a:'old thick new interesting', v:'read open write borrow', p:'shelf library desk bag', isa:'a thing made of pages'},
    music: {a:'soft loud beautiful sad', v:'play hear write love', p:'room concert radio head', isa:'sound in time'},
    song: {a:'soft old beautiful sad', v:'sing hear write learn', p:'radio room concert night', isa:'a piece of music'},
    friend: {a:'kind true old close', v:'help call trust meet', p:'school life work street', isa:'a person you trust'},
    computer: {a:'fast small old useful', v:'use fix build program', p:'desk office room lab', isa:'a machine'},
    robot: {a:'small quiet helpful metal', v:'build program help teach', p:'lab room factory story', isa:'a machine that thinks'},
    brain: {a:'small curious busy tired', v:'think learn rest grow', p:'head computer story', isa:'the thing that thinks'},
    time: {a:'short long slow quick', v:'pass wait spend save', p:'clock life day', isa:'a measure of change'},
    love: {a:'deep quiet true strong', v:'feel give share grow', p:'heart life story', isa:'a feeling'},
    hope: {a:'small quiet strong bright', v:'keep find lose hold', p:'heart night dark morning', isa:'a feeling'},
    fear: {a:'deep dark cold sudden', v:'feel face hide overcome', p:'heart night chest dark', isa:'a feeling'},
    dream: {a:'strange quiet deep bright', v:'have chase remember forget', p:'sleep night mind', isa:'a story while sleeping'},
    coffee: {a:'hot dark strong bitter', v:'drink brew pour sip', p:'cup kitchen shop morning', isa:'a drink'},
    tea: {a:'hot green warm light', v:'drink brew pour steep', p:'cup pot garden morning', isa:'a drink'},
    bread: {a:'warm fresh soft brown', v:'bake cut eat share', p:'kitchen oven table shop', isa:'a food'},
    rice: {a:'white warm soft plain', v:'cook boil eat serve', p:'bowl pot kitchen plate', isa:'a food'},
    pizza: {a:'hot cheesy round fresh', v:'eat bake share slice', p:'oven box table shop', isa:'a food'},
    house: {a:'small old warm quiet', v:'build clean leave enter', p:'street hill town garden', isa:'a place to live'},
    city: {a:'busy loud large bright', v:'visit leave cross build', p:'road map night hill', isa:'a place'},
    road: {a:'long narrow wide quiet', v:'walk cross follow build', p:'hill field town forest', isa:'a way between places'},
    light: {a:'bright soft warm golden', v:'shine glow fade turn', p:'window sky room lamp', isa:'energy you can see'},
    grass: {a:'green wet soft tall', v:'cut grow touch walk', p:'field garden park hill', isa:'a plant'},
    thunder: {a:'loud deep sudden distant', v:'hear rumble shake follow', p:'sky storm night cloud', isa:'a sound from the sky'},
    school: {a:'busy old loud quiet', v:'attend leave build teach', p:'town street morning road', isa:'a place to learn'},
    car: {a:'fast small old red', v:'drive fix park wash', p:'road garage street city', isa:'a machine'},
    bicycle: {a:'light quick old rusty', v:'ride fix pedal steer', p:'road street park hill', isa:'a machine with two wheels'},
    star: {a:'bright distant small golden', v:'see count watch wish', p:'sky night space window', isa:'a ball of burning gas'},
    cloud: {a:'white soft dark thin', v:'watch drift cover pass', p:'sky hill morning sea', isa:'water in the air'},
    river: {a:'wide slow cold deep', v:'cross follow swim fish', p:'valley forest bridge town', isa:'moving water'},
    forest: {a:'deep dark green quiet', v:'walk explore cross protect', p:'hill mountain valley mist', isa:'a place of many trees'},
    summer: {a:'hot long bright dry', v:'swim travel rest play', p:'sea beach garden road', isa:'a season'},
    winter: {a:'cold long dark quiet', v:'freeze sleep rest wait', p:'snow fire house window', isa:'a season'},
    morning: {a:'early quiet bright cold', v:'wake rise start begin', p:'sun window bed kitchen', isa:'a time of day'},
    night: {a:'quiet dark long cold', v:'sleep rest watch dream', p:'moon bed window sky', isa:'a time of day'},
    heart: {a:'warm kind strong tired', v:'beat love feel break', p:'chest body life story', isa:'a part of the body'},
    hand: {a:'warm small strong soft', v:'hold write touch help', p:'arm body pocket glove', isa:'a part of the body'},
    eye: {a:'bright tired blue sharp', v:'see watch look cry', p:'face head light window', isa:'a part of the body'},
    voice: {a:'soft loud warm quiet', v:'hear speak raise lower', p:'room throat night phone', isa:'a sound you make'},
    memory: {a:'old bright quiet deep', v:'keep lose hold share', p:'mind heart head story', isa:'a thing you remember'},
    truth: {a:'hard simple cold quiet', v:'tell find seek hide', p:'story word heart court', isa:'what is really so'},
    world: {a:'wide old strange small', v:'change see leave know', p:'space sea map story', isa:'everything around us'},
    earth: {a:'old round green small', v:'protect live leave see', p:'space sky sea ground', isa:'the planet we live on'}
  };
  var CGENERIC = {a:'small quiet strange simple', v:'see find use keep', p:'world mind story', isa:'a thing'};

  var CTEMPL = {
    sentence: ['the {n} is {a} and it makes me think of the {p}',
               'here is one the {a} {n} rests in the {p} and waits',
               'i can try the {n} is {a} and the {p} is never far away'],
    describe: ['{n} is {isa} that is {a} and often found in the {p}',
               'it is {isa} it is {a} and you can {v} there in the {p}'],
    explain: ['{n} is {isa} it means you can {v} there and it feels {a}',
              'the idea is simple {n} is {a} and it lives in the {p}',
              '{n} is {isa} the way to {v} is simple and it feels {a}'],
    story: ['once there was a {a} {n} who learned to {v} in the {p}',
            'let me try one day a {n} began to {v} and the {p} grew {a}'],
    opinion: ['i like the {n} because it is {a} and it can {v}',
              'i think the {n} is {a} it makes the {p} feel alive'],
    compare: ['the {n} is {a} while the {p} stays quiet it is a matter of scale',
              'one is {a} the other is not they only share the {p}'],
    howto: ['first find the {n} then {v} slowly until it is {a}',
            'start at the {p} take the {n} and {v} with care'],
    default: ['the {n} is {a} and it belongs in the {p}',
              'i think of the {a} {n} and the way it can {v}']
  };

  function cwords(t) { return (t.toLowerCase().match(/[a-z]+/g) || []); }

  function cextract(text) {
    var w = cwords(text), intent = 'default', i;
    for (i = 0; i < w.length; i++) if (CINTENT[w[i]]) { intent = CINTENT[w[i]]; break; }
    if (w.indexOf('sentence') >= 0 || w.indexOf('sentences') >= 0) intent = 'sentence';
    var skip = {}, k;
    for (i = 0; i < CSTOP.length; i++) skip[CSTOP[i]] = 1;
    for (i = 0; i < CVERBS.length; i++) skip[CVERBS[i]] = 1;
    for (k in CINTENT) skip[k] = 1;
    var body = [];
    for (i = 0; i < w.length; i++) if (!skip[w[i]] && w[i].length > 2) body.push(w[i]);
    return [intent, body];
  }

  // 介词后的词优先当主体（"story ABOUT moon" 的主体是 moon 不是 story）
  function csubject(text, body) {
    if (!body.length) return null;
    var w = cwords(text), i, j, cand;
    var preps = ['about', 'with', 'for', 'of', 'on'];
    for (j = 0; j < preps.length; j++) {
      var li = w.lastIndexOf(preps[j]);
      if (li >= 0 && li + 1 < w.length) {
        cand = w[li + 1];
        if (CNOUNS[cand]) return cand;
        if (cand.length > 2 && cand.substr(-2) === 'es' && CNOUNS[cand.slice(0, -2)]) return cand.slice(0, -2);
        if (cand.length > 1 && cand.substr(-1) === 's' && CNOUNS[cand.slice(0, -1)]) return cand.slice(0, -1);
      }
    }
    for (i = 0; i < body.length; i++) if (CNOUNS[body[i]]) return body[i];
    for (i = 0; i < body.length; i++) {
      if (body[i].substr(-2) === 'es' && CNOUNS[body[i].slice(0, -2)]) return body[i].slice(0, -2);
      if (body[i].substr(-1) === 's' && CNOUNS[body[i].slice(0, -1)]) return body[i].slice(0, -1);
    }
    var best = body[0];
    for (i = 1; i < body.length; i++) if (body[i].length > best.length) best = body[i];
    return best;
  }

  function compose(text) {
    var r = cextract(text), intent = r[0], body = r[1];
    if (!body.length) return null;
    var t = csubject(text, body);
    if (!t) return null;
    // 生造词不硬拼
    if (!CNOUNS[t] && !TOK.w2i[t]) return null;
    var d = CNOUNS[t] || CGENERIC;
    function pick(k) { var arr = d[k].split(' '); return arr[Math.floor(Math.random() * arr.length)]; }
    var tpl = CTEMPL[intent] || CTEMPL['default'];
    var s = tpl[Math.floor(Math.random() * tpl.length)];
    return s.replace('{n}', t).replace('{a}', pick('a')).replace('{v}', pick('v'))
            .replace('{p}', pick('p')).replace('{isa}', d.isa);
  }

  // 意图明确的长句优先拼装（模型在这类句式上常"自信地胡说"）
  function tryCompose(text) {
    if (!api.composeMode) return null;
    var r = cextract(text), intent = r[0], body = r[1];
    var first = ['sentence', 'describe', 'story', 'howto', 'explain'];
    if (first.indexOf(intent) >= 0 && body.length) return compose(text);
    return null;
  }

  function gen(text, n, temp, eosPenalty) {
    n = n || 40; temp = temp === undefined ? 0.0 : temp;
    eosPenalty = eosPenalty === undefined ? 6.0 : eosPenalty;
    // 算式交给计算器，不经过模型（模型算不准算术）
    var mm = tryMath(text);
    if (mm !== null) {
      return { text: mm, source: 'math', sim: null, matched_q: null, ids: [] };
    }
    // 中文避让：见中文直接开喷，不检索、不走模型
    if (api.zhMode && isChinese(text)) {
      return { text: ZH_SNARK[Math.floor(Math.random() * ZH_SNARK.length)],
               source: 'zh_dodge', sim: null, matched_q: null, ids: [] };
    }
    // 检索兜底（先剥前缀/归一缩写，再检索）
    if (RET) {
      var r = retrieveBest(text);
      if (r[0] !== null) return { text: r[0], source: 'retrieve', sim: r[1], matched_q: r[2], ids: [] };
    }
    var cp = tryCompose(text);
    if (cp) return { text: cp, source: 'compose', sim: null, matched_q: null, ids: [] };
    var ctx = TOK.tok('q ' + text + ' a');
    if (!ctx.length) ctx = [UNK];
    var CTX = CFG.CTX, o = [], step, ids, p, tid, lp, pr, i, s;
    var letterRun = 0;
    for (step = 0; step < n; step++) {
      ids = ctx.slice(-CTX);
      while (ids.length < CTX) ids.unshift(PAD);
      p = forwardLast(ids);
      // 低置信度早停
      if (step === 0) {
        var ent = 0, mx = 0;
        for (i = 0; i < p.length; i++) { if (p[i] > mx) mx = p[i]; if (p[i] > 1e-12) ent -= p[i] * Math.log(p[i]); }
        if (ent > 3.5 || mx < 0.50) return { text: '', source: 'model', sim: null, matched_q: null, ids: o };
      }
      // EOS 抑制（log 空间）
      if (eosPenalty > 0) {
        var lgA = new Float64Array(p.length), mx2 = -Infinity;
        for (i = 0; i < p.length; i++) {
          lgA[i] = Math.log(p[i] + 1e-12);
          if (i === EOS) lgA[i] -= (step < 4 ? eosPenalty : eosPenalty * 0.35);
          if (lgA[i] > mx2) mx2 = lgA[i];
        }
        s = 0;
        for (i = 0; i < p.length; i++) { lgA[i] = Math.exp(lgA[i] - mx2); s += lgA[i]; }
        for (i = 0; i < p.length; i++) p[i] = lgA[i] / s;
      }
      // 连续单字母防护
      if (letterRun >= 2) {
        for (i = 2048; i < p.length; i++) p[i] = 0;
        s = 0;
        for (i = 0; i < p.length; i++) s += p[i];
        if (s <= 1e-12) break;
        for (i = 0; i < p.length; i++) p[i] /= s;
      }
      if (temp <= 0.05) {
        tid = 0;
        for (i = 1; i < p.length; i++) if (p[i] > p[tid]) tid = i;
      } else {
        lp = new Float64Array(p.length); mx = -Infinity;
        for (i = 0; i < p.length; i++) { lp[i] = Math.log(p[i] + 1e-12) / Math.max(0.1, temp); if (lp[i] > mx) mx = lp[i]; }
        pr = new Float64Array(p.length); s = 0;
        for (i = 0; i < p.length; i++) { pr[i] = Math.exp(lp[i] - mx); s += pr[i]; }
        var rr = Math.random() * s, acc = 0; tid = p.length - 1;
        for (i = 0; i < p.length; i++) { acc += pr[i] / s; if (rr <= acc) { tid = i; break; } }
      }
      if (tid === EOS || tid >= B.total) break;
      // 打转检测
      if (step >= 5) {
        var tail = o.slice(-6), cnt = {}, spin = false;
        for (i = 0; i < tail.length; i++) { cnt[tail[i]] = (cnt[tail[i]] || 0) + 1; if (cnt[tail[i]] >= 3) spin = true; }
        if (spin) break;
      }
      letterRun = (tid >= 2048 && (TOK.rev[tid] || '').length === 1) ? letterRun + 1 : 0;
      o.push(tid); ctx.push(tid);
    }
    var res = TOK.merge(o);
    if (api.composeMode && res.split(/\s+/).length < 2) {
      var c2 = compose(text);
      if (c2) return { text: c2, source: 'compose', sim: null, matched_q: null, ids: o };
    }
    return { text: res, source: 'model', sim: null, matched_q: null, ids: o };
  }

  var api = {
    zhMode: true,          // 中文避让开关，网页里可 BINAC.zhMode=false 关闭
    composeMode: true,     // 长句拆解拼装开关
    load: load,
    gen: gen,
    ask: function (text, temp) { return gen(text, 40, temp === undefined ? 0.0 : temp); },
    tok: function (t) { return TOK.tok(t); },
    get brain() { return B; },
    get cfg() { return CFG; },
    get qaCount() { return RET && B.qa ? B.qa.length : 0; }
  };

  root.BINAC = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : this);
