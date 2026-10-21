import {Streamlit, type RenderData} from 'streamlit-component-lib';

type ApiReply={requestId:string;status:number;payload:unknown};
type ApiRequest={requestId:string;url:string;method:string;body?:string};
type Pending={request:ApiRequest;resolve:(response:Response)=>void};

if(window.parent!==window){
 const originalFetch=window.fetch.bind(window);
 const queue:Pending[]=[];
 let active:Pending|undefined;
 const sendNext=()=>{
  if(active||!queue.length)return;
  active=queue.shift();
  Streamlit.setComponentValue({kind:'api-request',...active!.request});
 };

 window.fetch=((input:RequestInfo|URL,init?:RequestInit)=>{
  const url=new URL(typeof input==='string'?input:input instanceof URL?input.href:input.url,window.location.href);
  if(!url.pathname.startsWith('/api/'))return originalFetch(input,init);
  return new Promise<Response>(resolve=>{
   const body=init?.body==null?undefined:typeof init.body==='string'?init.body:String(init.body);
   queue.push({request:{requestId:crypto.randomUUID(),url:url.pathname+url.search,method:init?.method||'GET',body},resolve});
   sendNext();
  });
 }) as typeof window.fetch;

 Streamlit.events.addEventListener(Streamlit.RENDER_EVENT,(event:Event)=>{
  const data=(event as CustomEvent<RenderData<{apiReply?:ApiReply}>>).detail;
  const reply=data.args.apiReply;
  if(reply&&active?.request.requestId===reply.requestId){
   active.resolve(new Response(JSON.stringify(reply.payload),{status:reply.status,headers:{'Content-Type':'application/json'}}));
   active=undefined;
   sendNext();
  }
  Streamlit.setFrameHeight();
 });
 Streamlit.setComponentReady();
 new ResizeObserver(()=>Streamlit.setFrameHeight()).observe(document.body);
}

void import('./main');
