package napi.main.exec.cpu;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import napi.main.exec.NVExecutable;


public class NThreadPool implements Runnable
{
	//private NThreader[] Runnable;
	private Iterable<NThreadable> Pool;

	private Thread[] T;
	private BigInteger BackwardScope = BigInteger.ONE;
	private BigInteger Counter = BigInteger.ZERO;
	private BigInteger ForwardScope = BigInteger.ONE;
		
	public NThreadPool(LinkedHashMap<NVExecutable,NThreadable> List)
	{
		Pool = (Iterable<NThreadable>) List;//new NThreader[List.size()];
		T = new Thread[List.size()];
		
		int[] TCounter= {0};
		List.forEach(
		(k,current)->
		{
			//Runnable[TCounter[0]] = new NThreader(current, TCounter[0], false); 
			T[TCounter[0]] = new Thread(current);
		    TCounter[0]++;  
		});
	}
	public NThreadPool(LinkedHashMap<NVExecutable,NThreadable> List, BigInteger forwardScope, BigInteger backwardScope)
	{
		BackwardScope=backwardScope;
		ForwardScope=forwardScope;
		Counter=BigInteger.ZERO;
		
		Pool = (Iterable<NThreadable>) List;//new NThreader[List.size()];
		T = new Thread[List.size()];
		
		int[] TCounter= {0};
		List.forEach(
		(k,current)->
		{
			//Runnable[TCounter[0]] = new NThreader(current, TCounter[0], false); 
			T[TCounter[0]] = new Thread(current);
		    TCounter[0]++;  
		});
	}
	public NThreadPool(ArrayList<NVExecutable> List, BigInteger forwardScope, BigInteger backwardScope)
	{

		
		BackwardScope=backwardScope;
		ForwardScope=forwardScope;
		Counter=BigInteger.ZERO;
		
		ArrayList<NThreadable> newPool = new ArrayList<NThreadable>();
		T = new Thread[List.size()];
		
		int[] TCounter= {0};
		List.forEach(
		(current)->
		{
			newPool.add(new NThreadable(current, TCounter[0], false)); 
			T[TCounter[0]] = new Thread(newPool.get(TCounter[0]));
		    TCounter[0]++;  
		});
		Pool = newPool;
	}	
	public NThreadPool(ArrayList<NVExecutable> List, int backwardScope)
	{
		this.BackwardScope=BigInteger.valueOf(backwardScope);
		Counter=BigInteger.ZERO;
		
		ArrayList<NThreadable> newRunnable = new ArrayList<NThreadable>();
		T = new Thread[List.size()];
		
		int[] TCounter= {0};
		List.forEach(
		(current)->
		{
			newRunnable.add(new NThreadable(current, TCounter[0], false)); 
			T[TCounter[0]] = new Thread(newRunnable.get(TCounter[0]));
		    TCounter[0]++;  
		});
		Pool = newRunnable;
	}	
	public NThreadPool(LinkedList<NVExecutable> List, int backwardScope)
	{
		this.BackwardScope=BigInteger.valueOf(backwardScope);
		Counter=BigInteger.ZERO;
		
		ArrayList<NThreadable> newPool = new ArrayList<NThreadable>();
		T = new Thread[List.size()];
		
		int[] TCounter= {0};
		List.forEach(
		(current)->
		{
			newPool.add(new NThreadable(current, TCounter[0], false)); 
			T[TCounter[0]] = new Thread(newPool.get(TCounter[0]));
		    TCounter[0]++;  
		});
		Pool = newPool;
	}	
	
	
	public void setBackward(BigInteger backward) {this.BackwardScope = backward;}
	public void setForwrad(BigInteger forward) {this.ForwardScope = forward;}
//=========================================================================================================================		
	private void redefineThreads() {
		int TCounter=0;
		Iterator<NThreadable> iterator = Pool.iterator();
		while(iterator.hasNext()) 
		{
			NThreadable current = iterator.next();
			T[TCounter] = new Thread(current);
			TCounter++;
		}

	}
//=========================================================================================================================		
	public void run()
	{		
		    Counter=BigInteger.ZERO;
		    //try {Thread.sleep(1000);} catch (InterruptedException e) {	e.printStackTrace();}
		    while(Counter.compareTo(ForwardScope)==-1) //HistoryCounter < History
			{
		    	//joinChildren();redefineThreads();
		    	startForwardThreading();redefineThreads();joinChildren();
		    	Counter=Counter.add(BigInteger.ONE);
			}	
			joinChildren();//necessary?
			Counter=BigInteger.ZERO;
		    System.out.println("Forward propagation successful!");
			while(Counter.compareTo(BackwardScope)==-1) //HistoryCounter < History
			{
			 //joinChildren();redefineThreads();
			 startBackwardThreading();redefineThreads();joinChildren();
			 Counter=Counter.add(BigInteger.ONE);
			}	
			joinChildren();//necessary?
			try {Thread.sleep(10);} catch (InterruptedException e) {	e.printStackTrace();}
			System.out.println("ThreadCoordinator: Workload finished!");			
	}
//=========================================================================================================================	
	private void joinChildren() 
	{
		for(int NCounter=0; NCounter<T.length; NCounter++)
		{
			try 
			{T[NCounter].join();System.out.println("joined");} 
			catch (InterruptedException e) 
			{
				System.out.println("Join exception!");
				e.printStackTrace();
			}
		}
	}

	private void startForwardThreading()
	{
		Iterator<NThreadable> iterator = Pool.iterator();
		while(iterator.hasNext()) 
		{
			NThreadable current = iterator.next();
			current.setIsBackpropagating(false);
			current.getExecutable().cleanup();
		}
		
		for(int TCounter=0; TCounter<T.length; TCounter++)
		{T[TCounter].start();}
		
	}

	//=========================================================================================================================
	private void startBackwardThreading()//true false - threading or not!
	{
		
		Iterator<NThreadable> iterator = Pool.iterator();
		while(iterator.hasNext()) 
		{
			NThreadable current = iterator.next();
			current.setIsBackpropagating(true);
			current.getExecutable().preBackward(Counter);
		}
		
		for(int TCounter=0; TCounter<T.length; TCounter++)
		{T[TCounter].start();}
		// try {Thread.sleep(10);} catch (InterruptedException e) {e.printStackTrace();}
	}	
	//=========================================================================================================================	
	
	
	
}



