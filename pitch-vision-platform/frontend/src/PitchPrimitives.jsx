import {useEffect,useState} from 'react';

export function useData(url){
  const [state,setState]=useState({loading:true,data:null,error:null});
  useEffect(()=>{
    const abort=new AbortController();
    setState({loading:true,data:null,error:null});
    fetch(url,{signal:abort.signal})
      .then(async response=>{
        if(!response.ok)throw Error(response.status===404?'This match is not available.':'Match data could not be loaded.');
        return response.json();
      })
      .then(data=>setState({data,loading:false,error:null}))
      .catch(error=>{if(error.name!=='AbortError')setState({data:null,loading:false,error:error.message});});
    return()=>abort.abort();
  },[url]);
  return state;
}

export function PitchLines({length=105,width=68}){
  const penaltyWidth=40.32,boxY=(width-penaltyWidth)/2;
  return <g fill="none" stroke="currentColor" strokeWidth=".3"><rect x=".15" y=".15" width={length-.3} height={width-.3}/><path d={`M${length/2} 0V${width}`}/><circle cx={length/2} cy={width/2} r="9.15"/><circle cx={length/2} cy={width/2} r=".25" fill="currentColor"/><path d={`M0 ${boxY}H16.5V${boxY+penaltyWidth}H0 M${length} ${boxY}H${length-16.5}V${boxY+penaltyWidth}H${length} M0 ${(width-18.32)/2}H5.5V${(width+18.32)/2}H0 M${length} ${(width-18.32)/2}H${length-5.5}V${(width+18.32)/2}H${length}`}/></g>;
}
