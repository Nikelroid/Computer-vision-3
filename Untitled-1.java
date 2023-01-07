import java.util.*;

 public class Graph {
     private int V;

     private ArrayList<ArrayList<Integer> > adj;

     Graph(int v) {
         V = v;
         adj = new ArrayList<ArrayList<Integer> >(v);
         for (int i = 0; i < v; ++i)
             adj.add(new ArrayList<Integer>());
     }

     void addEdge(int v, int w) { adj.get(v).add(w); }
     void topologicalSortUtil(int v, boolean visited[],
                              Stack<Integer> stack)
     {
         visited[v] = true;
         Integer i;

         Iterator<Integer> it = adj.get(v).iterator();
         while (it.hasNext()) {
             i = it.next();
             if (!visited[i])
                 topologicalSortUtil(i, visited, stack);
         }

         stack.push(new Integer(v));
     }

     void topologicalSort()
     {
         Stack<Integer> stack = new Stack<Integer>();

         boolean visited[] = new boolean[V];
         for (int i = 0; i < V; i++)
             visited[i] = false;

         for (int i = 0; i < V; i++)
             if (visited[i] == false)
                 topologicalSortUtil(i, visited, stack);
           while (stack.empty() == false){
               stack.pop();}
       /*  if (stack.empty() == false)
             System.out.print("1");
         else System.out.print("0");*/
     }
 }
        public class Untitled-1{
     public static void main(String[] args) {
         Scanner scan = new Scanner(System.in);
         int n = scan.nextInt();
         int m = scan.nextInt();
         Graph g = new Graph(n);
         for (int i = 0; i < m; i++) {
             int temp1 = scan.nextInt();
             int temp2 = scan.nextInt();
             g.addEdge(temp2,temp1);
         }
        try {
            g.topologicalSort();
            System.out.print("1");
        }
        catch (IndexOutOfBoundsException e) {
            System.out.print("0");
        }
}
     }


 



