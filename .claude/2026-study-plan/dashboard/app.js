/* =====================================================================
   APP LOGIC — rendering + progress engine + file sync.
   Reads plan data from the globals defined in data.js (loaded first).
   Edit the plan in data.js and the styling in styles.css; this file is
   the machinery and rarely needs to change.
   ===================================================================== */
(function(){
  "use strict";
  var LS="mlplan.v1.";
  var store={};
  try{ for(var i=0;i<localStorage.length;i++){var k=localStorage.key(i); if(k.indexOf(LS)===0){ store[k.slice(LS.length)]=localStorage.getItem(k);} } }catch(e){}
  function get(k){ return store[k]; }
  function getB(k){ return store[k]==="1"; }
  function set(k,v){ store[k]=v; try{ localStorage.setItem(LS+k, v); }catch(e){} }
  function del(k){ delete store[k]; try{ localStorage.removeItem(LS+k);}catch(e){} }

  // ---------- helpers ----------
  function el(tag,cls,html){var e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e;}
  function reqKeys(w){var ks=[]; w.core.forEach(function(_,i){ks.push("wk."+w.id+".c."+i);}); ks.push("wk."+w.id+".d"); return ks;}
  function weekDone(w){var ks=reqKeys(w),d=0; ks.forEach(function(k){if(getB(k))d++;}); return {done:d,total:ks.length};}
  // Weeks are stored 1..24 internally (stable progress keys) but DISPLAYED 0-indexed: W0..W23.
  function wnum(id){ return id-1; }
  function wlab(id){ var n=id-1; return "W"+(n<10?"0":"")+n; }

  // current week from today
  function currentIndex(){
    var now=new Date(); var idx=0;
    for(var i=0;i<WEEKS.length;i++){ if(new Date(WEEKS[i].start+"T00:00:00")<=now) idx=WEEKS[i].id; }
    return idx; // 0 => before start
  }
  var CUR=currentIndex();
  var CURW=CUR===0?1:CUR;

  // ---------- render: run cells ----------
  var cellsEl=document.getElementById("cells");
  var blkOrder=["A","B","C","D"];
  blkOrder.forEach(function(b){
    var wrap=el("div","blk");
    WEEKS.filter(function(w){return w.blk===b;}).forEach(function(w){
      var c=el("button","cell"); c.type="button";
      c.setAttribute("data-week",w.id);
      c.title=wlab(w.id)+" — "+w.title;
      c.setAttribute("aria-label",c.title);
      wrap.appendChild(c);
    });
    cellsEl.appendChild(wrap);
  });
  cellsEl.addEventListener("click",function(e){
    var c=e.target.closest(".cell"); if(!c)return;
    var id=+c.getAttribute("data-week");
    var wk=document.getElementById("week-"+id);
    if(wk){ openWeek(wk,true); wk.scrollIntoView({behavior:"smooth",block:"center"}); }
  });

  // ---------- render: this week ----------
  function itemRow(w,type,idx,item,checkedKey){
    var key=checkedKey;
    var li=el("li","item"); if(getB(key))li.classList.add("done");
    var cb=el("input"); cb.type="checkbox"; cb.className="chk"; cb.setAttribute("data-k",key); cb.checked=getB(key);
    cb.id="cb-"+key.replace(/\./g,"-");
    var tag=el("span","tag "+item.t, TYPE[item.t]||"DO");
    var lab=el("label"); lab.setAttribute("for",cb.id); lab.innerHTML=item.x;
    li.appendChild(cb); li.appendChild(tag); li.appendChild(lab);
    return li;
  }
  function delRow(w){
    var key="wk."+w.id+".d";
    var li=el("li","item"); if(getB(key))li.classList.add("done");
    var cb=el("input"); cb.type="checkbox"; cb.className="chk"; cb.setAttribute("data-k",key); cb.checked=getB(key);
    cb.id="cb-"+key.replace(/\./g,"-");
    var tag=el("span","tag proj","DELIV");
    var lab=el("label"); lab.setAttribute("for",cb.id); lab.innerHTML="<span class='del'>Deliverable:</span> "+w.del;
    li.appendChild(cb); li.appendChild(tag); li.appendChild(lab);
    return li;
  }
  function buildWeekBody(w){
    var frag=document.createDocumentFragment();
    frag.appendChild(el("div","grp-lab","Core — must do"));
    var ulc=el("ul","items");
    w.core.forEach(function(item,i){ ulc.appendChild(itemRow(w,"c",i,item,"wk."+w.id+".c."+i)); });
    frag.appendChild(ulc);
    if(w.stretch&&w.stretch.length){
      frag.appendChild(el("div","grp-lab","Stretch — if ahead"));
      var uls=el("ul","items");
      w.stretch.forEach(function(txt,i){
        var key="wk."+w.id+".s."+i;
        var li=el("li","item"); if(getB(key))li.classList.add("done");
        var cb=el("input"); cb.type="checkbox"; cb.className="chk"; cb.setAttribute("data-k",key); cb.checked=getB(key); cb.id="cb-"+key.replace(/\./g,"-");
        var tag=el("span","tag","+");
        var lab=el("label"); lab.setAttribute("for",cb.id); lab.innerHTML=txt;
        li.appendChild(cb); li.appendChild(tag); li.appendChild(lab); uls.appendChild(li);
      });
      frag.appendChild(uls);
    }
    var uld=el("ul","items"); uld.style.marginTop="10px"; uld.appendChild(delRow(w)); frag.appendChild(uld);
    if(w.note){ frag.appendChild(el("div","note","⚠ "+w.note)); }
    return frag;
  }

  var tw=WEEKS[CURW-1];
  var twWrap=document.getElementById("thisweek");
  (function(){
    var card=el("div","card now-card");
    var top=el("div","nc-top");
    top.appendChild(el("span","nc-code",wlab(tw.id)));
    top.appendChild(el("span","nc-title",tw.title));
    top.appendChild(el("span","nc-dates mono",tw.dates));
    top.appendChild(el("span","nc-badge", CUR===0?"STARTS SOON":"BLOCK "+tw.blk));
    card.appendChild(top);
    var body=el("div","pad");
    body.appendChild(buildWeekBody(tw));
    card.appendChild(body);
    twWrap.appendChild(card);
  })();

  // today string
  (function(){
    var now=new Date();
    var m=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    document.getElementById("today-str").textContent=m[now.getMonth()]+" "+now.getDate()+", "+now.getFullYear();
  })();

  // ---------- render: timeline blocks ----------
  var blocksEl=document.getElementById("blocks");
  BLOCKS.forEach(function(b){
    var block=el("div","block"); block.id="block-"+b.id;
    var head=el("div","block-head");
    head.appendChild(el("span","block-tag",b.id));
    var meta=el("div"); meta.appendChild(el("div","bh-title",b.title)); meta.appendChild(el("div","bh-meta",b.meta));
    head.appendChild(meta);
    var prog=el("div","block-prog");
    var bar=el("div","bar"); var fill=el("i"); fill.id="bfill-"+b.id; bar.appendChild(fill);
    prog.appendChild(bar); var bp=el("span","bp"); bp.id="bpct-"+b.id; bp.textContent="0%"; prog.appendChild(bp);
    head.appendChild(prog);
    block.appendChild(head);

    WEEKS.filter(function(w){return w.blk===b.id;}).forEach(function(w){
      var wk=el("div","week"); wk.id="week-"+w.id;
      if(w.id===CURW) wk.classList.add("is-now");
      var hb=el("button","wk-head"); hb.type="button"; hb.setAttribute("aria-expanded","false");
      hb.appendChild(el("span","wk-code",wlab(w.id)));
      hb.appendChild(el("span","wk-title",w.title));
      if(w.id===CURW) hb.appendChild(el("span","nowpill","NOW"));
      hb.appendChild(el("span","wk-dates mono",w.dates));
      var mini=el("span","wk-mini"); var mbar=el("div","bar"); var mfill=el("i"); mfill.id="wfill-"+w.id; mbar.appendChild(mfill); mini.appendChild(mbar);
      hb.appendChild(mini);
      hb.appendChild(el("span","wk-chev","›"));
      wk.appendChild(hb);
      var bodyWrap=el("div","wk-body");
      bodyWrap.appendChild(buildWeekBody(w));
      wk.appendChild(bodyWrap);
      hb.addEventListener("click",function(){ toggleWeek(wk); });
      block.appendChild(wk);
    });
    blocksEl.appendChild(block);
  });
  function toggleWeek(wk){ var open=wk.classList.toggle("open"); wk.querySelector(".wk-head").setAttribute("aria-expanded",open?"true":"false"); }
  function openWeek(wk,force){ if(force){ wk.classList.add("open"); wk.querySelector(".wk-head").setAttribute("aria-expanded","true"); } }
  // open current week by default in timeline
  (function(){ var wk=document.getElementById("week-"+CURW); if(wk) openWeek(wk,true); })();

  // ---------- render: projects ----------
  function projCard(p,isRepro){
    var card=el("div","card proj"+(p.anchor?" anchor":""));
    var head=el("div","p-head");
    head.appendChild(el("span","p-id",p.id?p.id:p.name.split(" ")[0]));
    var nm=el("div"); nm.appendChild(el("div","p-name",p.name)); nm.appendChild(el("div","p-tag",p.tag)); head.appendChild(nm);
    if(p.anchor) head.appendChild(el("span","anchor-flag","ANCHOR"));
    card.appendChild(head);
    var body=el("div","p-body");
    if(!isRepro) body.appendChild(el("div",null,"<span style='font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);font-weight:600'>"+p.feeds+"</span>"));
    body.appendChild(el("p",null,p.body));
    if(p.metrics) body.appendChild(el("div","metrics","<b>Metrics:</b> "+p.metrics));
    card.appendChild(body);
    var key=(isRepro?"repro.":"proj.")+p.id;
    var row=el("div","deploy-row"); if(getB(key))row.classList.add("on");
    var cb=el("input"); cb.type="checkbox"; cb.className="chk"; cb.setAttribute("data-k",key); cb.checked=getB(key); cb.id="cb-"+key.replace(/\./g,"-"); cb.style.width="18px"; cb.style.height="18px";
    var lab=el("label"); lab.setAttribute("for",cb.id); lab.textContent=isRepro?"Reproduction packaged & written up":"Deployed with live URL + numbers";
    row.appendChild(cb); row.appendChild(lab);
    card.appendChild(row);
    return card;
  }
  var pg=document.getElementById("projects-grid");
  PROJECTS.forEach(function(p){ pg.appendChild(projCard(p,false)); });
  var rg=document.getElementById("repros-grid");
  REPROS.forEach(function(p){ rg.appendChild(projCard(p,true)); });

  // ---------- render: checkpoints ----------
  var cpEl=document.getElementById("cp-list");
  CPS.forEach(function(c){
    var d=el("div","cp"+(c.big?" big":""));
    d.appendChild(el("div","cp-when","CHECKPOINT "+c.n+" · "+c.when));
    d.appendChild(el("h4",null,c.title));
    d.appendChild(el("div","cp-q","“"+c.q+"”"));
    var g=el("div","cp-grades");
    g.appendChild(el("span","pill g","🟢 continue"));
    g.appendChild(el("span","pill y","🟡 tighten / fix"));
    g.appendChild(el("span","pill r","🔴 re-cut, don't quit"));
    d.appendChild(g);
    cpEl.appendChild(d);
  });

  // ---------- render: posts ----------
  var postsEl=document.getElementById("posts-list");
  POSTS.forEach(function(p){
    var key="post."+p.n;
    var li=el("li","item"); if(getB(key))li.classList.add("done");
    var cb=el("input"); cb.type="checkbox"; cb.className="chk"; cb.setAttribute("data-k",key); cb.checked=getB(key); cb.id="cb-"+key.replace(/\./g,"-");
    var tag=el("span","tag","#"+p.n);
    var lab=el("label"); lab.setAttribute("for",cb.id); lab.innerHTML="<span class='mono' style='color:var(--ink-3);font-size:11px'>"+p.w+"</span> &nbsp;"+p.t;
    li.appendChild(cb); li.appendChild(tag); li.appendChild(lab);
    postsEl.appendChild(li);
  });

  // ---------- parking lot ----------
  var park=document.getElementById("parking");
  park.value=get("parking.text")||"";
  park.addEventListener("input",function(){ set("parking.text",park.value); scheduleSave(); });

  // ---------- progress engine ----------
  function updateProgress(){
    var totAll=0,doneAll=0;
    var byBlk={A:{d:0,t:0},B:{d:0,t:0},C:{d:0,t:0},D:{d:0,t:0}};
    WEEKS.forEach(function(w){
      var r=weekDone(w); totAll+=r.total; doneAll+=r.done;
      byBlk[w.blk].t+=r.total; byBlk[w.blk].d+=r.done;
      // week mini
      var mf=document.getElementById("wfill-"+w.id); if(mf) mf.style.width=(r.total?Math.round(r.done/r.total*100):0)+"%";
      // cell state
      var cell=cellsEl.querySelector('.cell[data-week="'+w.id+'"]');
      if(cell){
        cell.classList.remove("done","part","now");
        if(r.total&&r.done===r.total) cell.classList.add("done");
        else if(r.done>0){ cell.classList.add("part"); cell.style.setProperty("--f",Math.round(r.done/r.total*100)+"%"); }
        if(w.id===CURW) cell.classList.add("now");
      }
    });
    var pct=totAll?Math.round(doneAll/totAll*100):0;
    document.getElementById("run-pct").textContent=pct+"%";
    document.getElementById("kpi-timeline").innerHTML=pct+"<small>%</small>";
    blkOrder.forEach(function(b){
      var o=byBlk[b], p=o.t?Math.round(o.d/o.t*100):0;
      var f=document.getElementById("bfill-"+b); if(f)f.style.width=p+"%";
      var bp=document.getElementById("bpct-"+b); if(bp)bp.textContent=p+"%";
    });
    // KPIs proj/posts
    var pd=0; PROJECTS.forEach(function(p){ if(getB("proj."+p.id))pd++; });
    document.getElementById("kpi-proj").innerHTML=pd+"<small> / 6</small>";
    var pp=0; POSTS.forEach(function(p){ if(getB("post."+p.n))pp++; });
    document.getElementById("kpi-posts").innerHTML=pp+"<small> / 12</small>";
    // now
    var w=WEEKS[CURW-1];
    document.getElementById("run-now").innerHTML=(CUR===0?"Starts "+w.dates:"<b>Week "+wnum(w.id)+"</b> / 24 · Block "+w.blk);
    document.getElementById("kpi-now").textContent=wlab(w.id)+" / 24";
    document.getElementById("kpi-now-sub").textContent="Block "+w.blk+" · "+w.dates;
  }

  // ---------- delegated change handling ----------
  document.addEventListener("change",function(e){
    var cb=e.target;
    if(cb&&cb.classList&&cb.classList.contains("chk")){
      var k=cb.getAttribute("data-k");
      if(cb.checked) set(k,"1"); else del(k);
      // toggle done styling on nearest .item / .deploy-row
      var item=cb.closest(".item"); if(item) item.classList.toggle("done",cb.checked);
      var dr=cb.closest(".deploy-row"); if(dr) dr.classList.toggle("on",cb.checked);
      // sync duplicate checkboxes with same key (this-week vs timeline)
      var dupes=document.querySelectorAll('.chk[data-k="'+k.replace(/"/g,'')+'"]');
      dupes.forEach(function(d){ if(d!==cb){ d.checked=cb.checked; var di=d.closest(".item"); if(di)di.classList.toggle("done",cb.checked); var dd=d.closest(".deploy-row"); if(dd)dd.classList.toggle("on",cb.checked);} });
      updateProgress();
      scheduleSave();
    }
  });

  // reset
  document.getElementById("reset").addEventListener("click",function(e){
    e.preventDefault();
    if(!confirm("Reset all progress and the parking lot in this browser? The plan itself is untouched.")) return;
    Object.keys(store).forEach(function(k){ try{localStorage.removeItem(LS+k);}catch(e){} });
    store={};
    document.querySelectorAll(".chk").forEach(function(c){ c.checked=false; var i=c.closest(".item"); if(i)i.classList.remove("done"); var d=c.closest(".deploy-row"); if(d)d.classList.remove("on"); });
    park.value="";
    updateProgress();
    if(connected) scheduleSave();
  });

  // ---------- save & sync to a file ----------
  var handle=null, saveTimer=null, connected=false, lastTs="";
  var LSMETA="mlplan.meta.savedAt";
  var hasFS = ('showSaveFilePicker' in window) && ('showOpenFilePicker' in window);
  var isHttp = /^https?:$/.test(location.protocol);
  function localSavedAt(){ try{ return localStorage.getItem(LSMETA)||""; }catch(e){ return ""; } }
  function markSaved(ts){ lastTs=ts||lastTs; try{ localStorage.setItem(LSMETA, lastTs); }catch(e){} }
  // decide whether an incoming file should overwrite what's in this browser
  function fileIsNewer(obj){
    var f=obj&&obj.savedAt?obj.savedAt:"";
    if(Object.keys(store).length===0) return true;   // fresh browser → take the file
    if(!f) return false;
    var l=localSavedAt(); if(!l) return true;
    return f>l;                                       // ISO timestamps sort lexically
  }
  function payload(){ lastTs=new Date().toISOString(); return JSON.stringify({_:"ml-plan-progress",v:1,savedAt:lastTs,data:store},null,2); }
  function syncDom(){
    document.querySelectorAll(".chk").forEach(function(c){
      var k=c.getAttribute("data-k"); c.checked=getB(k);
      var i=c.closest(".item"); if(i)i.classList.toggle("done",c.checked);
      var d=c.closest(".deploy-row"); if(d)d.classList.toggle("on",c.checked);
    });
    if(park) park.value=get("parking.text")||"";
    updateProgress();
  }
  function applyState(obj){
    var data=obj&&obj.data?obj.data:obj;
    if(!data||typeof data!=="object") throw new Error("bad file");
    Object.keys(store).slice().forEach(function(k){ del(k); });
    Object.keys(data).forEach(function(k){ set(k,String(data[k])); });
    if(obj&&obj.savedAt) markSaved(obj.savedAt);
    syncDom();
  }
  function setStatus(msg){
    var s=document.getElementById("sync-status"), rc=document.getElementById("sync-reconnect");
    if(!s) return;
    if(connected&&handle){ s.innerHTML="<span class='dot on'></span> Auto-saving to <b>"+(handle.name||"progress.json")+"</b> — no clicks needed"; if(rc)rc.hidden=true; }
    else if(msg==="detected"){ s.innerHTML="<span class='dot on'></span> Detected <b>progress.json</b> in this folder (read-only). Click <b>Save to file</b> once to enable auto-save."; }
    else if(msg==="downloaded"){ s.innerHTML="<span class='dot'></span> Saved <b>progress.json</b> to your Downloads — move it next to dashboard.html, then commit it"; }
    else if(msg==="loaded"){ s.innerHTML="<span class='dot on'></span> Loaded into this browser"; }
    else if(msg==="error"){ s.innerHTML="<span class='dot err'></span> File error — using this browser only for now"; }
    else{ s.innerHTML="<span class='dot'></span> Saving in this browser only — click <b>Save to file</b> to keep a progress.json"; }
  }
  async function writeHandle(){
    if(!handle)return;
    try{ var body=payload(); var w=await handle.createWritable(); await w.write(body); await w.close(); markSaved(); setStatus(); }
    catch(e){ connected=false; setStatus("error"); }
  }
  function scheduleSave(){ if(!connected||!handle)return; clearTimeout(saveTimer); saveTimer=setTimeout(writeHandle,500); }
  function idb(cb){ try{ var r=indexedDB.open("mlplan-fs",1); r.onupgradeneeded=function(){ r.result.createObjectStore("h"); }; r.onsuccess=function(){ cb(r.result); }; r.onerror=function(){ cb(null); }; }catch(e){ cb(null); } }
  function saveHandle(h){ idb(function(db){ if(!db)return; try{ db.transaction("h","readwrite").objectStore("h").put(h,"progress"); }catch(e){} }); }
  function loadHandleRef(cb){ idb(function(db){ if(!db){cb(null);return;} try{ var g=db.transaction("h","readonly").objectStore("h").get("progress"); g.onsuccess=function(){cb(g.result||null);}; g.onerror=function(){cb(null);}; }catch(e){cb(null);} }); }
  function downloadFallback(){
    var blob=new Blob([payload()],{type:"application/json"}), url=URL.createObjectURL(blob), a=document.createElement("a");
    a.href=url; a.download="progress.json"; document.body.appendChild(a); a.click();
    setTimeout(function(){ URL.revokeObjectURL(url); a.remove(); },120); setStatus("downloaded");
  }
  function fileInputFallback(){
    var inp=document.createElement("input"); inp.type="file"; inp.accept="application/json,.json";
    inp.onchange=function(){ var f=inp.files&&inp.files[0]; if(!f)return; var rd=new FileReader();
      rd.onload=function(){ try{ applyState(JSON.parse(rd.result)); connected=false; setStatus("loaded"); }catch(e){ alert("That isn't a valid progress.json."); } };
      rd.readAsText(f); };
    inp.click();
  }
  async function doSave(){
    if(hasFS){ try{
      handle=await window.showSaveFilePicker({suggestedName:"progress.json",types:[{description:"Progress JSON",accept:{"application/json":[".json"]}}]});
      connected=true; saveHandle(handle); await writeHandle(); setStatus();
    }catch(e){ if(e&&e.name==="AbortError")return; downloadFallback(); } }
    else downloadFallback();
  }
  async function doLoad(){
    if(hasFS){ try{
      var picked=await window.showOpenFilePicker({types:[{description:"Progress JSON",accept:{"application/json":[".json"]}}]});
      handle=picked[0]; var f=await handle.getFile(); applyState(JSON.parse(await f.text()));
      connected=true; saveHandle(handle); setStatus();
    }catch(e){ if(e&&e.name==="AbortError")return; fileInputFallback(); } }
    else fileInputFallback();
  }
  async function reconnect(){
    if(!handle)return;
    try{ var perm=await handle.requestPermission({mode:"readwrite"}); if(perm!=="granted"){ setStatus(); return; }
      var f=await handle.getFile(); var txt=await f.text(); if(txt&&txt.trim()){ var o=JSON.parse(txt); if(fileIsNewer(o)) applyState(o); }
      connected=true; scheduleSave(); setStatus();
    }catch(e){ setStatus("error"); }
  }
  // On load: silently re-attach the file if we can, otherwise auto-detect a committed progress.json in this folder.
  async function autoRestore(){
    // 1) Chrome/Edge: reuse the file handle from last time — silent if permission still granted
    if(hasFS){
      var h=await new Promise(function(res){ loadHandleRef(res); });
      if(h){
        handle=h;
        var perm="prompt";
        try{ perm=await h.queryPermission({mode:"readwrite"}); }catch(e){}
        if(perm==="granted"){
          try{
            var f=await h.getFile(); var txt=await f.text();
            connected=true;
            if(txt&&txt.trim()){ var obj=JSON.parse(txt); if(fileIsNewer(obj)) applyState(obj); else scheduleSave(); }
            else scheduleSave();
            setStatus(); return;
          }catch(e){ connected=false; }
        } else {
          var rc=document.getElementById("sync-reconnect"); if(rc) rc.hidden=false;
        }
      }
    }
    // 2) Served over http(s): detect the committed progress.json next to dashboard.html (read-only)
    if(isHttp){
      try{
        var r=await fetch("progress.json",{cache:"no-store"});
        if(r&&r.ok){ var o=await r.json(); if(fileIsNewer(o)) applyState(o); setStatus("detected"); return; }
      }catch(e){}
    }
    setStatus();
  }
  (function initSync(){
    var sv=document.getElementById("sync-save"), ld=document.getElementById("sync-load"), rc=document.getElementById("sync-reconnect");
    if(sv) sv.addEventListener("click",doSave);
    if(ld) ld.addEventListener("click",doLoad);
    if(rc) rc.addEventListener("click",reconnect);
    autoRestore();
  })();

  updateProgress();
})();

