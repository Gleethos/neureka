import java.util.Random;

import napi.utility.NMessageFrame;

public class GridGame 
{
	NMessageFrame gameDisplay; 
	int X, Y;
	int px, py;
	
	int[][] map;
	int     currentReward;
	
	int[]   rewardHistory;
	int[]   actionHistory;
	int[][] mapHistory;
	int     historyLength = 15;
	
	double loss = 0;
	double historyValue;
	
	GridGame(int Xdim, int Ydim)
	{
		X = Xdim; Y = Ydim; gameDisplay = new NMessageFrame("The Grid Game"); 
		map = new int[X][Y];  
		mapHistory = new int[historyLength][X*Y]; 
		rewardHistory = new int[historyLength];  
		actionHistory = new int[historyLength]; 
		boolean playerIsSet = false, 
		goalIsSet = false;
		int tileID=0;
		Random dice = new Random(); 
		
		for(int Yid=0; Yid<X; Yid++) 
	    {
			for(int Xid=0; Xid<X; Xid++) 
			{
				tileID=dice.nextInt(5)+1; 
				if(tileID>3) 
				{
					if(playerIsSet==false)
					{
						map[Yid][Xid]=-3; 
						playerIsSet=true; 
						py=Yid; px=Xid;
					}  
	                else
	                {
	                	if(goalIsSet==false&(Yid==Y-2))
	                 	{
	                		map[Yid][Xid]=3;
	                		goalIsSet=true;
	                	}        
	                    else
	                    {
	                    	map[Yid][Xid]=0;
	                    }
	                }
				} 
				else 
				{
					if(tileID==2) 
					{
						map[Yid][Xid]=-1;
					}
					else
					{
						map[Yid][Xid]=+1;
					}
				}
			}
		}
	}                  
	public void restartGame()
	{
		boolean playerSet = false, goalSet = false; int tileID=0;
		Random dice = new Random(); 
		for(int Yid=0; Yid<X; Yid++) 
		{	
			for(int Xid=0; Xid<X; Xid++) 
			{
				tileID=dice.nextInt(5)+1; 
				if(tileID>3) 
				{
					if(playerSet==false)
					{
						map[Yid][Xid]=-3; playerSet=true; py=Yid; px=Xid;
					}  
	                else
	                {
	                	if(goalSet==false&(Yid==Y-2))
	                	{
	                		map[Yid][Xid]=3;
	                		goalSet=true; 
	                	}        
	                    else
	                    {
	                    	map[Yid][Xid]=0;
	                    }
	                }
				} 
				else 
				{
					if(tileID==2) 
					{
						map[Yid][Xid]=-1;
					}
					else
					{
						map[Yid][Xid]=+1;
					}
				}
			}
		}
		rewardHistory[0]=0; 
		mapHistory[0]=getCurrentMapArray(); 
		actionHistory[0]=0;
	}	
	public int   getHistoryAction() 
	{
		return actionHistory[historyLength-1];
	}
	public int[] getHistoryMap() 
	{
		return mapHistory[historyLength-1];
	}
	public int[] getHistoryView() //might not right
	{
		int counter=0;
		int x=0; 
		int y=0; 
		int[][] map = new int[Y][X];
		for(int Yid=0; Yid<X; Yid++) 
	    {
			for(int Xid=0; Xid<X; Xid++) 
	        {
				if(mapHistory[(historyLength-1)][counter]==-3) 
				{
					y=Yid; 
					x=Xid;
				} 
				map[Yid][Xid]=mapHistory[(historyLength-1)][counter];
				counter++;
	        }
	    }
		int[] view = new int[4];
		// #0#
		// 1#2
		// #3#
		for(int Yid=0; Yid<X; Yid++) 
	    {
			for(int Xid=0; Xid<X; Xid++) 
	        {
				if(y==Yid && x==Xid) 
	            {
					if     (y==0  ) {view[0]=map[Y-1][ x ]; view[3]=map[ 1 ][x]; }
					else if(y==X-1) {view[0]=map[Y-2][ x ]; view[3]=map[ 0 ][x];}
					else            {view[0]=map[y-1][ x ]; view[3]=map[y+1][x];}
					if     (x==0  ) {view[1]=map[ y ][X-1]; view[2]=map[ y ][1];}
					else if(x==X-1) {view[1]=map[ y ][X-2]; view[2]=map[ y ][0]; }
					else            {view[1]=map[ y ][x-1]; view[2]=map[ y ][x+1]; }        	  
	             }
				counter++;
	   	    }
	    }	
		return view;
	}
	
	public void printHistoricAgentView() 
	{
		int[] v = getHistoryView();
		gameDisplay.print("#"+v[0]+"#\n");	
		gameDisplay.print(v[1]+"A"+v[2]+"\n");	
		gameDisplay.print("#"+v[3]+"#\n action: "+getHistoryAction()+"\n\n");	
	}
	
	public int getHistoryKeyID() {return 1;}
	public double getHistoryValue() {return historyValue;}
	
	public void printMap() 
	{
		for(int Yid=0; Yid<Y; Yid++) 
	    {
			gameDisplay.print("   |");
			for(int Xid=0; Xid<X; Xid++) 
			{
				gameDisplay.print(getMapCharOf(map[Yid][Xid]));
			}
			gameDisplay.print("|");gameDisplay.println("");
		}
	    gameDisplay.println("Reward: "+currentReward+" Value: "+historyValue+" py:"+py+" px:"+px+"\n");
	}
	
	public void printMap(int time) 
	{
		for(int Yid=0; Yid<Y; Yid++) 
	    {
			gameDisplay.print("   |");
			for(int Xid=0; Xid<X; Xid++) 
			{	
				gameDisplay.print(getMapCharOf(mapHistory[mapHistory.length-time][X*Xid+Yid]));
			}
			gameDisplay.print("|");gameDisplay.println("");
		}
	     gameDisplay.println("Reward: "+currentReward+" Value: "+historyValue+" py:"+py+" px:"+px+"\n");
	}
	public void printMapHistory(double valueEstimate)
	{
		/*
			int counter=0;
			for(int Yid=0; Yid<Y; Yid++) 
			{gameDisplay.print("   |");for(int Xid=0; Xid<X; Xid++) {gameDisplay.print(getMapCharOf(mapHistory[historyLength-1][counter]));counter++;}gameDisplay.print("|");gameDisplay.println("");}
		*/
		gameDisplay.println("Historic Reward: "+rewardHistory[historyLength-1]+" Historic Value: "+historyValue+" Value Estimation: "+valueEstimate);
		gameDisplay.println("Historic Action: "+actionHistory[historyLength-1]+" ");
		printHistoricAgentView();
	}
	private String getMapCharOf(int tileID) 
	{
		switch(tileID) 
		{
			case -1: return ":#:";
			case -3: return ":A:";
			case  0: return ":O:";
			case  1: return ":H:";
			case  2: return ":X:";
			case  3: return "|||||";
			case  4: return ":P:";
			default: return "| # |";
		}
	}
	public void printInGameConsole(String message) 
	{
		gameDisplay.print(message);
	}	
	public int[] getCurrentMapArray() 
	{
		int[] array = new int[(X*Y)]; 
		int counter =0;
		for(int Yid=0; Yid<X; Yid++)
		{
			for(int Xid=0; Xid<X; Xid++) 
			{
				array[counter]=map[Yid][Xid]; 
				counter++;
			}
		}
		return array;
	}
	private void calculateNextFrame() 
	{
		if(map[py][px]==0) {currentReward= 0;}
		if(map[py][px]==1) {currentReward= 1;}
		if(map[py][px]==-1){currentReward=-1;}
		if(map[py][px]==3) {currentReward= 3;}
		map[py][px]=-3;  
		storeMapState(); 
		calculateNewValue(); 
	}
	private void storeMapState()
	{
		for(int Hid=historyLength-2; Hid>=0; Hid--)//This is stupid!
		{
			rewardHistory[Hid+1]=rewardHistory[Hid]; 
			mapHistory[Hid+1]=mapHistory[Hid]; 
			actionHistory[Hid+1]=actionHistory[Hid]; 
		}
		rewardHistory[0]=currentReward; 
		mapHistory[0]=getCurrentMapArray();
	}
	private void calculateNewValue() 
	{
		historyValue=0;
		for(int Tid=0; Tid<rewardHistory.length-1; Tid++) 
		{
			historyValue += Math.pow(loss, rewardHistory.length-Tid-2)*rewardHistory[Tid]; 
		}
	 }
	public void moveUp() {map[py][px]=0; py--; if(py<0) {py=Y-1;} gameDisplay.print("moved: UP(-2); \n");calculateNextFrame(); actionHistory[1]=-2;}
	
	public void moveDown() {map[py][px]=0; py++; if(py>=Y){py=0; }gameDisplay.print("moved: DOWN(-1); \n"); calculateNextFrame(); actionHistory[1]=-1;}
	
	public void moveRight() {map[py][px]=0; px++; if(px>=X) {px=0; }gameDisplay.print("moved: RIGHT(1); \n"); calculateNextFrame(); actionHistory[1]=1;}
	
	public void moveLeft() {map[py][px]=0; px--; if(px<0) {px=X-1; }gameDisplay.print("moved: LEFT(2); \n"); calculateNextFrame(); actionHistory[1]=2;}
	
	public void moveNowhere()
	{
		gameDisplay.print("moved: Nowhere; \n"); 
		calculateNextFrame(); 
		actionHistory[1]=0;
	}
	public void move(int direction) 
	{
		if(direction==0) {moveUp();}
		if(direction==1) {moveRight();}
		if(direction==2) {moveLeft();}
		if(direction==3) {moveDown();}
	}
	public void makeRandomMove() 
	{
		Random dice = new Random(); int moveID= dice.nextInt(4); 
		switch(moveID) 
		{
			case 0: moveUp();break; 
			case 1: moveDown();break; 
			case 2: moveRight();break; 
			case 3: moveLeft(); break;
		}
	}


	
}
