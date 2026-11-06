import importlib.util, sys, copy, json
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('pdb',__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','pdbench.py')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
P=m.PROTO
class A: server_log=None
def row(role,c,rep,**kw):
    r=dict(concurrency=c,rep=rep,num_prompts=max(80,10*c),completed=max(80,10*c),mean_input_tokens=84840,mean_output_tokens=1 if role=='p' else 1200,
           mean_ttft_ms=4400.0,p50_ttft_ms=4000.0,p99_ttft_ms=5000.0,qpm=54.0,steady_qpm=55.0,prefix_hit=0.93 if role=='p' else 1.0,
           _out_lens=[1]*80 if role=='p' else [P['d_output_len']]*max(80,10*c),_errors=[],_warmups=10 if role=='p' else 2)
    if role=='d': r.update(mean_tpot_ms=15.0,p99_tpot_ms=16.0,mtp_accept=None)
    r.update(kw); return r
M={"protocol_version":"pdbench-1.0","attest":{"cold_compile_cache_cleared":True,"exclusive_machine":True},"endpoints":{"hit_rate_source":"metrics"}}
def run(role,raw,man=M,fill=None,tiers=None):
    ck=m.make_checks(role,raw,None,A(),man,fill or dict(num_prompts=200,completed=200),tiers or sorted({r['concurrency'] for r in raw}),P['expected_fingerprint'])
    return {c['id']:c['status'] for c in ck}
good=[row('p',4,1),row('p',4,2)]
base=run('p',good); print('good  ',base); assert all(v in('PASS','SKIP') for v in base.values()),base
def expect(name,role,raw,cid,want,**kw):
    got=run(role,raw,**kw)[cid]; print(f'{name:28} {cid}={got}'); assert got==want,(name,cid,got,want)
expect('too few requests','p',[row('p',4,1,num_prompts=30,completed=30),row('p',4,2)],'C4','FAIL')
expect('a failed request','p',[row('p',4,1,completed=79),row('p',4,2)],'C3','FAIL')
expect('hit rate too low','p',[row('p',4,1,prefix_hit=0.5),row('p',4,2)],'C5','FAIL')
expect('hit rate missing','p',[row('p',4,1,prefix_hit=None),row('p',4,2)],'C5','FAIL')
expect('P repeat reused the same data (hit rate too high)','p',[row('p',4,1),row('p',4,2,prefix_hit=0.9999)],'C5','FAIL')
expect('input length off target','p',[row('p',4,1,mean_input_tokens=60000),row('p',4,2)],'C6','FAIL')
expect('P output truncated by EOS','p',[row('p',4,1,_out_lens=[1]*79+[0]),row('p',4,2)],'C7','FAIL')
expect('warmup count mismatch','p',[row('p',4,1,_warmups=0),row('p',4,2)],'C8','FAIL')
expect('only one run','p',[row('p',4,1)],'C9','FAIL')
expect('2% spread warns','p',[row('p',4,1,qpm=54.0),row('p',4,2,qpm=55.1)],'C9','WARN')
expect('5% spread blocks','p',[row('p',4,1,qpm=54.0),row('p',4,2,qpm=57.0)],'C9','FAIL')
expect('no manifest','p',good,'C1','SKIP',man=None)
expect('cold compile not declared','p',good,'C8','FAIL',man={**M,'attest':{'cold_compile_cache_cleared':False,'exclusive_machine':True}})
dgood=[row('d',4,1),row('d',4,2)]
expect('D output truncated','d',[row('d',4,1,_out_lens=[300]*80),row('d',4,2)],'C7','FAIL')
def runf(raw):
    ck=m.make_checks('d',raw,None,A(),M,dict(num_prompts=200,completed=200),[4],P['expected_fingerprint'])
    return {c['id']:c['status'] for c in ck}
assert runf([row('d',4,1,_out_lens=[1200]*40),row('d',4,2,_out_lens=[1200]*40)])['C7']=='PASS'
assert runf([row('d',4,1,_out_lens=[P['d_output_len']]*39+[P['d_output_len']-1]),row('d',4,2,_out_lens=[1200]*40)])['C7']=='FAIL', "with a fixed length, even one EOS-truncated request must FAIL"
assert runf([row('d',4,1,_out_lens=None),row('d',4,2,_out_lens=[1200]*40)])['C7']=='FAIL', "not being able to read per-request lengths must not PASS"
assert P['d_output_len']==1200, 'protocol: the D replay emits a fixed 1200 output tokens'
print('C7 per-request check of the fixed D output length   OK')
expect('D replay hit rate 1.0 passes','d',dgood,'C5','PASS')
expect('D hit rate 0.9','d',[row('d',4,1,prefix_hit=0.9),row('d',4,2)],'C5','FAIL')
expect('cache fill incomplete','d',dgood,'C7f','FAIL',fill=dict(num_prompts=200,completed=150))
cp=m.capacity('p',[dict(concurrency=c,verdict=v,qpm=q,steady_qpm=q) for c,v,q in [(4,'PASS',10.0),(6,'PASS',12.0),(7,'FAIL',11.0)]])
assert cp['max_qpm']['concurrency']==6 and cp['first_fail_concurrency']==7, cp
print('P capacity = highest-QPM tier within SLO c=6, first fail c=7   OK')
cd=m.capacity('d',[dict(concurrency=c,verdict=v,qpm=q,steady_qpm=q) for c,v,q in [(4,'PASS',10.0),(8,'PASS',20.0),(10,'FAIL',19.0),(12,'PASS',30.0),(13,'FAIL',29.0)]])
assert cd['contiguous']['concurrency']==8 and cd['max_qpm']['concurrency']==12 and cd['first_fail_concurrency']==10, cd
print('D non-monotonic: contiguous c=8 / max c=12 / first fail c=10   OK')
print('ALL TESTS PASSED')

# ---------------- pre-flight checks: hash canonicalization / server probe / the newer checks ----------------
assert m.canon_hash(b'{"a":1,"b":[1,2]}') == m.canon_hash(b'{ "b": [1, 2],\n "a": 1 }'), "the hash must be independent of key order and whitespace"
assert m.canon_hash(b'{"a":1}') != m.canon_hash(b'{"a":2}'), "different values must hash differently"
print('canon_hash                   OK')

srv, rec = m.stub_server(); url = f"http://127.0.0.1:{srv.server_address[1]}"
r = m.probe_server(url, "x"); print('probe (compliant stub)      ', r['ok'], r['detail'][:70]); assert r['ok']
assert len(rec) == 1 and json.loads(rec[0][1])['stream_options'] == {'include_usage': True}, "the probe request must carry stream_options.include_usage"
pb = json.loads(rec[0][1])
assert pb['stream'] is True and pb['ignore_eos'] is True and pb['temperature'] == 0.9 and pb['max_completion_tokens'] == 8, f"the probe request body is missing fields: {pb}"
assert json.loads(rec[0][1])['messages'][0]['content'][0]['type'] == 'text', "the probe must use the same segment-list content as the real client"
srv.shutdown()

import http.server, threading, json as _j
def serve(handler_body):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0))); self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            handler_body(self.wfile)
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start(); return s
chunk = lambda o: b"data: " + _j.dumps(o).encode() + b"\n\n"
s1 = serve(lambda w: (w.write(chunk({"choices": [{"delta": {"content": "a"}}]})), w.write(b"data: [DONE]\n\n")))   # no usage
r = m.probe_server(f"http://127.0.0.1:{s1.server_address[1]}", "x"); print('probe (no usage)            ', r['ok']); assert not r['ok'] and 'usage' in r['detail']; s1.shutdown()
s2 = serve(lambda w: (w.write(chunk({"choices": [{"delta": {"content": "a"}}]})), w.write(chunk({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})), w.write(b"data: [DONE]\n\n")))  # EOS ends the stream early
r = m.probe_server(f"http://127.0.0.1:{s2.server_address[1]}", "x"); print('probe (ignore_eos ignored)  ', r['ok']); assert not r['ok'] and 'ignore_eos' in r['detail']; s2.shutdown()
r = m.probe_server("http://127.0.0.1:1", "x"); print('probe (connection failed)   ', r['ok']); assert not r['ok']

ev_ok = dict(tokenizer_fp='abc', selftest=dict(ok=True, detail='ok', client_version='0.24.0'), probe=dict(ok=True, detail='ok'))
def runev(ev, role='p', raw=None, **kw):
    ck = m.make_checks(role, raw or good, None, A(), M, None if role == 'p' else dict(num_prompts=200, completed=200), [4], P['expected_fingerprint'], ev)
    return {c['id']: c['status'] for c in ck}
print('newer checks SKIP without evidence  ', {k: v for k, v in runev(None).items() if k in ('C2t', 'C13', 'C14')})
assert runev(None)['C13'] == 'SKIP' and runev(None)['C14'] == 'SKIP'
assert runev(ev_ok)['C13'] == 'PASS' and runev(ev_ok)['C14'] == 'PASS'
assert runev({**ev_ok, 'selftest': dict(ok=False, detail='hash mismatch')})['C13'] == 'FAIL'
assert runev({**ev_ok, 'probe': dict(ok=False, detail='no usage')})['C14'] == 'FAIL'
P_old = (m.PROTO['expected_tokenizer_fp'], m.PROTO['client_vllm_version'])
m.PROTO['expected_tokenizer_fp'] = 'abc'; assert runev(ev_ok)['C2t'] == 'PASS'; assert runev({**ev_ok, 'tokenizer_fp': 'zzz'})['C2t'] == 'FAIL'
m.PROTO['client_vllm_version'] = '0.24.0'; assert runev(ev_ok)['C13'] == 'PASS'
m.PROTO['client_vllm_version'] = '0.23.0'; assert runev(ev_ok)['C13'] == 'FAIL', "a client version other than the pinned one must FAIL"
m.PROTO['expected_tokenizer_fp'], m.PROTO['client_vllm_version'] = P_old

# Replay set evicted from the prefix cache: the tier's first run shows a lower hit rate and the second
# recovers, so the two repeats diverge. The old 0.98 lower bound let such runs through and only C9 caught
# them indirectly; with the bound raised to 0.999, C5 FAILs them directly.
assert P['d_hit_min'] == 0.999, "the D replay hit-rate lower bound must be 0.999"
expect('D replay evicted (hit 0.99)', 'd', [row('d', 10, 1, prefix_hit=0.99), row('d', 10, 2)], 'C5', 'FAIL')
expect('D replay evicted (hit 0.998)', 'd', [row('d', 14, 1, prefix_hit=0.998), row('d', 14, 2)], 'C5', 'FAIL')
assert runf([row('d', 4, 1, prefix_hit=1.0), row('d', 4, 2, prefix_hit=1.0)])['C5'] == 'PASS', "a full hit rate after the re-touch must PASS"

# Tier ceiling guard: a tier is capped by the replay-set size, so any tier whose 10*c exceeds it is bound to
# fail C4 and must be rejected before the run starts.
import os as _os, tempfile as _tf
_d = _tf.mkdtemp()
open(_os.path.join(_d, m.D_FILE), 'w').write(''.join('{"x":1}\n' for _ in range(200)))
def guard(tiers):   # calls the kit's real guard function rather than reimplementing its logic
    over, cap, cmax = m.d_tier_ceiling([int(x) for x in tiers.split()], _d)
    return over, cmax
over, cmax = guard('1 2 4 8 20')
print(f'tiers <=20 do not trip the guard      over={over} c_max={cmax}'); assert over == [] and cmax == 20
over, _ = guard('1 2 4 8 20 21 24 32')
print(f'tiers >20 trip the guard              over={over}'); assert over == [21, 24, 32]
open(_os.path.join(_d, m.D_FILE), 'w').write(''.join('{"x":1}\n' for _ in range(120)))   # a smaller replay set lowers the ceiling accordingly
over, cmax = guard('1 2 4 8 12 16')
print(f'ceiling with a 120-prompt replay set  over={over} c_max={cmax}'); assert over == [16] and cmax == 12
assert all(P['mult'] * c <= P['fill_n_max'] for c in [int(x) for x in m.DEFAULT_TIERS['d'].split()]), "the default D tiers must not exceed the ceiling the replay set can support"
print('default D tiers all within ceiling    ', m.DEFAULT_TIERS['d'])

# The fill size is derived from the target tiers: fill only what this sweep needs rather than maxing out the
# replay set (maxing it out raises the resident footprint and makes more tiers need a re-prefill).
open(_os.path.join(_d, m.D_FILE), 'w').write(''.join('{"x":1}\n' for _ in range(200)))
for tiers, want in [('1 2 4 6 8 10', 100), ('1 2 4 8 12 14 16 17 18 19 20', 200), ('1 2 4', 40), ('1 2 4 6 8 9 10 11 12 13 14 15 16 18', 180)]:
    got = m.d_fill_n([int(x) for x in tiers.split()], _d)
    print(f'highest tier {max(int(x) for x in tiers.split()):2d} -> fill {got:3d} prompts'); assert got == want, (tiers, got, want)

# C15: preemption counter
def runp(raw):
    ck = m.make_checks('d', raw, None, A(), M, dict(num_prompts=200, completed=200), [4], P['expected_fingerprint'])
    return {c['id']: c['status'] for c in ck}
assert runp([row('d', 4, 1, preemptions=0), row('d', 4, 2, preemptions=0)])['C15'] == 'PASS', 'zero preemptions must PASS'
assert runp([row('d', 4, 1, preemptions=3), row('d', 4, 2, preemptions=0)])['C15'] == 'FAIL', 'any preemption must FAIL'
assert runp([row('d', 4, 1), row('d', 4, 2)])['C15'] == 'SKIP', 'no preemption counter collected must SKIP'
print('C15 preemptions               0=PASS / >0=FAIL / not collected=SKIP')

print('ALL TESTS PASSED (pre-flight checks, replay eviction, tier ceiling guard, derived fill size, preemptions)')
