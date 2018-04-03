package graphdbInt;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;

import com.hp.hpl.jena.graph.Graph;
import com.hp.hpl.jena.graph.Triple;
import com.hp.hpl.jena.query.Dataset;
import com.hp.hpl.jena.query.ReadWrite;
import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.Property;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.rdf.model.Statement;
import com.hp.hpl.jena.tdb.TDBFactory;


public class getMLN {

	private static final String storage_dir = System.getProperty("user.dir");
	static boolean overwrite = false;
	public static ArrayList<clause> get(String dbName) throws IOException
	{
		String MLN = "./dataset/schema.db";
	
			ArrayList<clause> cls = new ArrayList<clause>();
			BufferedReader in = null;
			try {
				in = new BufferedReader(new FileReader(MLN));
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			String line;
			boolean active = false;
			while((line=in.readLine())!=null){
				if(active == true && !line.trim().isEmpty())
				{
					if(Character.isDigit(line.charAt(0)))
					{
						clause c = new clause();
						String[] lits = line.split("v");
						for(int i =0;i<lits.length;i++)
						{
							String[] tokens = lits[i].trim().split("\\(|,\\s*|\\)");
							
							String subj= tokens[1];
							String pred= tokens[0];
							String obj=(tokens.length > 2 ? tokens[2]:tokens[1]);
							
							//System.out.print(pred+":"+subj+","+obj+"|");
							
							if(i==0)
							{
								c.wt = Double.parseDouble(pred.split("  ")[0]);
								//System.out.println(pred.split("  ")[1]);
								c.preds.add(pred.split("  ")[1].trim().replace("!", "_")+":"+subj+","+obj);
							}
							else
								c.preds.add(pred.replace("!", "_")+":"+subj+","+obj);
						}
						System.out.print("\n");
						
						cls.add(c);
					}
				}
				if(line.contains("Rules"))
				{
					active = true;
				}
			}
			return cls;
	}
	public static Hashtable<String,ArrayList<String>> getPredArgtypes(String dbsc)
	{
		String MLN = dbsc;
		Hashtable<String,ArrayList<String>> argtypes = new Hashtable<String,ArrayList<String>>();
		BufferedReader in = null;
		try 
		{
			in = new BufferedReader(new FileReader(MLN));
		
		
			String line;
			while((line = in.readLine())!=null)
			{
				if(line.contains("Rules"))
					break;
				else
				{
					String s = line;
					if(!line.contains("//") && !line.trim().isEmpty())
					{
//						System.out.println(line);
						String[] arr = s.split("\\(|,\\s*|\\)");
						ArrayList<String> a = new ArrayList<String>();
						for(int i=1;i<arr.length;i++)
						{
							a.add(arr[i]);
						}
						argtypes.put(arr[0], a);
					}
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return argtypes;
	}
	
}
