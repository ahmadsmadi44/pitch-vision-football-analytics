import {useEffect,useRef} from 'react';

/**
 * Thin React adapter around dashersw/liquid-glass-js's MIT-licensed WebGL
 * Container. CSS remains the fallback when WebGL, motion, or capture is
 * unavailable.
 */
export default function LiquidGlass({as:Tag='div',className='',children,borderRadius=18,tintOpacity=.2}){
  const host=useRef(null);
  useEffect(()=>{
    if(!host.current||window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;
    const test=document.createElement('canvas');
    if(!test.getContext('webgl'))return;
    let instance,canvas,Container,cancelled=false;
    import('./vendor/liquid-glass/Container.js').then(module=>{
      if(cancelled||!host.current)return;
      Container=module.default;
      window.glassControls={...(window.glassControls||{}),blurRadius:7,edgeIntensity:.012,rimIntensity:.065,baseIntensity:.008,edgeDistance:.18,rimDistance:.85,baseDistance:.12,cornerBoost:.018,rippleEffect:.035};
      instance=new Container({borderRadius,tintOpacity,type:'rounded'});
      canvas=instance.canvas;
      canvas.className='liquid-glass-canvas';
      canvas.style.zIndex='0';
      canvas.style.pointerEvents='none';
      host.current.prepend(canvas);
      instance.element=host.current;
      instance.updateSizeFromDOM();
    }).catch(()=>{});
    return ()=>{
      cancelled=true;
      if(instance)instance.gl_refs.gl=null;
      canvas?.remove();
      if(Container&&instance)Container.instances=Container.instances.filter(item=>item!==instance);
    };
  },[borderRadius,tintOpacity]);
  return <Tag ref={host} className={`glass-container liquid-glass-surface ${className}`}><div className="liquid-glass-content">{children}</div></Tag>;
}
