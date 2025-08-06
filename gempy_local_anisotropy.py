import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

# for CI testing
smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.6')
pyro.set_rng_seed(1)



class grid:
    def __init__(self):
        super(grid, self).__init__()
        self.grid_data = []
        
    def Regular_grid(self, extent, resolution, dtype):
        
        dims = len(resolution)
        
        # Generate linspace per dimension
        for i in range(dims):
            start = extent[2*i]
            end = extent[2*i + 1]
            self.grid_data.append(torch.linspace(start, end, resolution[i], dtype=dtype))

        # Create meshgrid (cartesian product of all dimensions)
        self.mesh = torch.meshgrid(*self.grid_data, indexing="ij")  # Use 'ij' to keep dimensionality consistent

        # Flatten and stack each dimension
        flat = [m.reshape(-1) for m in self.mesh]
        full_grid = torch.stack(flat, dim=1)  # Shape: (num_points, num_dimensions)

        return full_grid
    
    def Custom_grid(self, custom_data):  
        return custom_data
        
        

    
class Gempy(grid):
    def __init__(self,project_name:str, extent:list, resolution:list):
        
        super(Gempy, self).__init__()
        self.project_name = project_name
        self.extent = extent
        self.resolution =resolution
        self.dtype = torch.float32
        self.grid_status={}
        self.sp_coord={}
        self.op_coord={}
        
        self.data={"Regular":None, "Custom":None}
        self.data["Regular"]= grid.Regular_grid(self, extent=self.extent, resolution=self.resolution, dtype=self.dtype)
        self.grid_status["Regular"]="Active"
        self.grid_status["Custom"] = None
        
        self.a_T = 5
        self.c_o_T = self.a_T**2/14/3
        self.s_1 = 0.01
        self.s_2 = 0.01
        
    def interpolation_options(self):
        print("a_T =", self.a_T)
        print("c_o_T =", self.c_o_T)
        print("s_1 =", self.s_1)
        print("s_2 =",self.s_2)
    def interpolation_options_set(self, a_T=5, c_o_T = 0.5952380952380952, s_1=0.01, s_2 = 0.01):
        self.a_T = a_T
        self.c_o_T = c_o_T
        self.s_1 = s_1
        self.s_2 = s_2
        
    def interface_data(self, sp_coord):
        print("\n ############# Gempy Options ############# \n")
        self.interpolation_options()
        self.sp_coord = sp_coord
        print("\n ############# Interface data ############# \n",self.sp_coord)
        
        
    def orientation_data(self, op_coord):
        self.op_coord = op_coord
        print("\n ############# Orientation data############# \n",self.op_coord)
        
    def activate_custom_grid(self, custom_grid_data):
        if custom_grid_data.shape[1]==len(self.resolution):
            self.data["Custom"] = grid.Custom_grid(self, custom_grid_data)
            self.grid_status["Custom"]="Active"
            
                  
    def active_grid(self):
        print("Grid status\n", self.grid_status)
        
    def deactivate_grid(self, grid_type:str):
        self.grid_status[grid_type]= None
        print("Grid status\n", self.grid_status)
        
    def activate_grid(self, grid_type:str):
        self.grid_status[grid_type]= "Active"
        #print("Grid status\n", self.grid_status)
        
    
    
    def covariance_function(self, r):
        r_by_at = r/self.a_T
        C_r = self.c_o_T *( 1 - 7 * (r_by_at)**2 + 8.75 * (r_by_at)**3 - 3.5 * (r_by_at)**5 + 0.75 * (r_by_at)**7)
        return C_r
    
    def first_derivative_covariance_function(self, r):
        C_r_dash =self.c_o_T *( - 14 * (r/self.a_T**2) + 105/4 * (r**2/self.a_T**3) - 35/2 * (r**4/self.a_T**5) + 21/4 * (r**6/self.a_T**7))
        # C_r_dash =self.c_o_T *( - 14 * (r_by_at)**2 + 105/4 * (r_by_at)**3 - 35/2 * (r_by_at)**5 + 21/4 * (r_by_at)**7)/ r
        return C_r_dash
    
    def first_derivative_covariance_function_divided_by_r(self, r):
        C_r_dash_by_r = self.c_o_T *( - 14 / ((self.a_T)**2) + 105/4 * (r/(self.a_T)**3) - 35/2 * (r**3/(self.a_T)**5) + 21/4 * (r**5/(self.a_T)**7))
        return C_r_dash_by_r
    
    def second_derivative_covariance_function(self,r):
        C_r_dash_dash =self.c_o_T * 7 * (9 * r ** 5 - 20 * self.a_T ** 2 * r ** 3 + 15 * self.a_T ** 4 * r - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)
        return C_r_dash_dash

    def squared_euclidean_distance(self, x_1,x_2, Transformation_matrix=torch.eye(2)):
        x_1 = x_1 @ Transformation_matrix.T
        x_2 = x_2 @ Transformation_matrix.T

        sqd = torch.sqrt(torch.clip(torch.reshape(torch.sum(x_1**2,1),shape =(x_1.shape[0],1))+\
        torch.reshape(torch.sum(x_2**2,1),shape =(1,x_2.shape[0]))-\
        2*( x_1@ x_2.T), min=0.0)+ 1e-12)
        
        return sqd

    def cartesian_dist_hu(self, x1, x2, Transformation_matrix=torch.eye(2) ):
        x1 = x1@ Transformation_matrix.T
        x2 = x2@ Transformation_matrix.T
        k = x1.shape[1]
        H =[]
        dummy_H = []
        # We converted x' = x A . 
        # There for delta x' = h_u (old) = delta x A
        for i in range(k):
            delta_x_i = x1[:,i] - torch.reshape(x2[:,i], shape=(x2.shape[0],1))
            H.append(delta_x_i)
        for i in range(k):
            a=torch.zeros(H[i].shape)
            for j in range(k):
                a= a +  Transformation_matrix[j,i] * H[j]
            dummy_H.append(a)
        H = dummy_H
        return H

    def cov_gradients(self, dist_tiled, H, Transformation_matrix=torch.eye(2), nugget_effect_grad=1/3):
        k = len(H) # component of gradient available
        n = int(H[0].shape[0]/ k) # number of place where gradient is defined

        #################################################################################################
        # For cross term of gradient 
        #################################################################################################
        # if we have many component of Gradient, we can club it to make , H =[h_u, h_v, h_w,....] 
        #H =[h_u, h_v] 
        C_G = torch.zeros((n*k, n*k))
        for i in range(len(H)):
            for j in range(len(H)):
                hu_hv = H[i][n*i:n*(i+1), n*j:n*(j+1)] * H[j][n*i:n*(i+1), n*j:n*(j+1)]
                dist =dist_tiled[n*i:n*(i+1), n*j:n*(j+1)]
                dist_sqr = dist **2 
                condition = dist_sqr!=0
                hu_hv_by_dist_sqr = torch.where(condition, hu_hv/ dist_sqr, 0.0)
                
                # Cross gradient term for C_ZuZv
                if i != j:
                    t2 = - self.first_derivative_covariance_function_divided_by_r(dist) + self.second_derivative_covariance_function(dist)
                    anisotrop_term = self.first_derivative_covariance_function_divided_by_r(dist) * torch.sum(Transformation_matrix[:,i]*Transformation_matrix[:,j])
                    array_test = -(hu_hv_by_dist_sqr * t2 + anisotrop_term)
                    for l in range(n):
                        for m in range(n):
                            C_G[i*n+l,j*n+m] = array_test[l,m]
                # Gradient for similar type of gradient
                else:
                    dist_cube = dist **3
                    condition = dist_cube!=0
                    hu_hv_by_dist_cube = torch.where(condition, hu_hv/dist_cube, 0.0)
                    
                    t4 = self.first_derivative_covariance_function(dist)
                    
                    # Here we have added the last term for anisotropy effect
                    t5 = torch.sum((Transformation_matrix[:,i])**2) * self.first_derivative_covariance_function_divided_by_r(dist) 
                    
                    t6 = self.second_derivative_covariance_function(dist)
                    
                    ################################# Note ############################################
                    # Covariance of gradient is negative for points closer to zero
                    ##################################################################################
            
                    array_test = -(- t4 * hu_hv_by_dist_cube + t5 +  t6 * hu_hv_by_dist_sqr )
                    for l in range(n):
                        for m in range(n):
                            C_G[i*n+l,j*n+m] = array_test[l,m]
        # If the distance is greater than a, then function C_z(r) = 0 . 
        # Therefore, we can replace the element of covariance matrix with 0 for distance greater than a
        condition_1 = dist_tiled<=self.a_T
        
        C_G = torch.where(condition_1, C_G, 0.0)
        C_G = C_G + nugget_effect_grad * torch.eye(n*k)

        return C_G

    def cov_interface(self, ref_layer_points,rest_layer_points, Transformation_matrix=torch.eye(2),nugget_effect_interface=1/3):
        sed_rest_rest = self.squared_euclidean_distance(rest_layer_points,rest_layer_points, Transformation_matrix=Transformation_matrix)
        sed_ref_rest = self.squared_euclidean_distance(ref_layer_points,rest_layer_points, Transformation_matrix=Transformation_matrix)
        sed_rest_ref = self.squared_euclidean_distance(rest_layer_points,ref_layer_points, Transformation_matrix=Transformation_matrix)
        sed_ref_ref = self.squared_euclidean_distance(ref_layer_points,ref_layer_points, Transformation_matrix=Transformation_matrix)
        
        C_I = self.covariance_function(sed_rest_rest) -\
            self.covariance_function(sed_ref_rest) -\
            self.covariance_function(sed_rest_ref) +\
            self.covariance_function(sed_ref_ref)

        return C_I + nugget_effect_interface * torch.eye(C_I.shape[0])

    ## Cartesian distance between dips and interface points

    def cartesian_dist_no_tile(self, x_1,x_2, Transformation_matrix=torch.eye(2)):
        x_1 = x_1 @ Transformation_matrix.T
        x_2 = x_2 @ Transformation_matrix.T
        k = x_1[0].shape[0]
        H_I = []
        for i in range(k):
            H_I.append(((x_1[:,i] - torch.reshape(x_2[:,i],[x_2.shape[0],1]))).T)
        
        Dummy_H_I=[]
        
        for i in range(k):
            a=torch.zeros(H_I[i].shape)
            for j in range(k):
                a= a + Transformation_matrix[j,i] * H_I[j]
            Dummy_H_I.append(a)
        
        H_I = Dummy_H_I
        H_I = torch.concat(H_I,axis=0)
        
        return H_I

    def cov_interface_gradients(self, hu_rest,hu_ref, Position_G_Modified, rest_layer_points, ref_layer_points, Transformation_matrix=torch.eye(2)):
        sed_dips_rest = self.squared_euclidean_distance(Position_G_Modified,rest_layer_points,Transformation_matrix=Transformation_matrix)
        sed_dips_ref = self.squared_euclidean_distance(Position_G_Modified,ref_layer_points,Transformation_matrix=Transformation_matrix)
        
        C_GI = - hu_rest * self.first_derivative_covariance_function_divided_by_r(sed_dips_rest) + hu_ref* self.first_derivative_covariance_function_divided_by_r(sed_dips_ref)
        
        return C_GI

    def set_rest_ref_matrix2(self, number_of_points_per_surface,input_position):
        ref_layer_points=[]
        rest_layer_points=[]
        for layer in input_position:
            ref_layer_points.append(layer[-1])
            rest_layer_points.append(layer[0:-1])
        # refrence points for each layer is repeated as number of non-referenced point
        
        repeats = number_of_points_per_surface-1

        ref_layer_points = torch.repeat_interleave(torch.stack(ref_layer_points ,axis = 0),repeats=repeats,axis = 0) 
        
        # Non referenced point. 
        rest_layer_points = torch.concat(rest_layer_points,axis = 0)
        return ref_layer_points,rest_layer_points
        
        
    def Ge_model(self, Transformation_matrix=None, nugget_effect_grad=1/3,nugget_effect_interface=1/3):
        ''' 
            Args:
                Input data:
                    input_position:     A list of list. Where each element of outer list is the information for each layer. 
                                        Inside each layer is the coordinates of position with last element is the refrence of that layer.
                    gradient_position:  A list of coordinates of gradient position
                    gradient_value:     Values of Gradient corresponding to the position mentioned in gradient_position
        '''
        # Assuming the there exist different layers in geological model. It can be assumed as scalar field. 
        
        self.number_of_layer = len(self.sp_coord)
        # Each layer containst the information of location of point where we have the information about scalar data. 
        # The last location is of the reference point for each layer.
        
        ## defining the dips position
        self.Position_G = self.op_coord["Positions"].to(self.dtype) # Location where Dips or gradient are given
        self.Value_G    = self.op_coord["Values"].to(self.dtype) # Gx, Gy, ..., Gk are the componet of gradient available at the given location
        n= self.Position_G.shape[0] # Total number of points available for gradient or dips
        k = self.Position_G[0].shape[0] # Total number of component available for the gradient
        # Since we have two component of the gradient, we can write the position twice corresponding to each coponent. We are assuming that 
        # Z is a scalar not a vector. Therefor we divide gradient into two different Z_u and Z_v
        
        Position_Add=[]
        for i in range(k):
            Position_Add.append(self.Position_G)
        self.Position_G_Modified = torch.cat(Position_Add, axis=0)
        
        number_of_points_per_surface=[]
        input_position=[]
        for keys, values in self.sp_coord.items():
            number_of_points_per_surface.append(len(values))
            input_position.append(values.to(self.dtype))

        self.number_of_points_per_surface = torch.tensor(number_of_points_per_surface)
        
        
        self.ref_layer_points,self.rest_layer_points = self.set_rest_ref_matrix2(self.number_of_points_per_surface,input_position)
        
        dist_position = self.squared_euclidean_distance(self.Position_G_Modified, self.Position_G_Modified, Transformation_matrix=Transformation_matrix) 
        # the dist_position can be all zero if there is only one point is defined where gradient is known.
        # 
        dist_position = dist_position #+  torch.eye(dist_position.shape[0])

        # Calculate the cartesian distance 

        H = self.cartesian_dist_hu(self.Position_G_Modified, self.Position_G_Modified, Transformation_matrix=Transformation_matrix)
        
        C_G = self.cov_gradients(dist_tiled=dist_position, H=H, Transformation_matrix=Transformation_matrix,nugget_effect_grad=nugget_effect_grad)
        
        C_I = self.cov_interface(self.ref_layer_points,self.rest_layer_points, Transformation_matrix=Transformation_matrix, nugget_effect_interface=nugget_effect_interface)
        
        hu_rest = self.cartesian_dist_no_tile(self.Position_G,self.rest_layer_points, Transformation_matrix=Transformation_matrix)
        hu_ref = self.cartesian_dist_no_tile(self.Position_G,self.ref_layer_points, Transformation_matrix=Transformation_matrix)   

        C_GI = self.cov_interface_gradients(hu_rest,hu_ref, self.Position_G_Modified, self.rest_layer_points, self.ref_layer_points, Transformation_matrix=Transformation_matrix)

        C_IG = C_GI.T

        K = torch.concat([torch.concat([C_G,C_GI],axis = 1),
        torch.concat([C_IG,C_I],axis = 1)],axis = 0)  

        # For kriging system in dual form require the list of all the Z. For gradient part, we can write the term but for Z(x)-Z(x0)=0 always.

        Modified_Value_G = self.Value_G #@ torch.eye(2)
        Modified_Value_G_flatten = torch.reshape((Modified_Value_G).T, [-1])
        b = torch.concat([Modified_Value_G_flatten,torch.zeros(K.shape[0]-Modified_Value_G_flatten.shape[0])],axis = 0)
        
        b = torch.reshape(b,shape = [b.shape[0],1])
        
        self.w = torch.linalg.solve(K,b)
    
    def Solution_grid(self, grid_coord, Transformation_matrix, section_plot= False):
        self.Ge_model(Transformation_matrix=Transformation_matrix)
        
        self.ref_points = torch.unique(self.ref_layer_points,dim=0)
        #print("grid_coord shape",grid_coord.shape)
        if grid_coord is not None:
            grid_data_plus_ref = torch.concat([grid_coord, self.ref_points],dim=0)
        else:
            grid_data_plus_ref = self.ref_points
            
        #print("grid_coord_plus shape",grid_data_plus_ref.shape)
        hu_Simpoints = self.cartesian_dist_no_tile(self.Position_G,grid_data_plus_ref,Transformation_matrix=Transformation_matrix)
        
        sed_dips_SimPoint = self.squared_euclidean_distance(self.Position_G_Modified,grid_data_plus_ref, Transformation_matrix=Transformation_matrix)
        

        ####################################### TODO #######################################
        # Check whether we need to transform first_derivative_covariance_function_divided_by_r 
        # by transformation matrix somehow
        ####################################################################################
        sigma_0_grad  =  self.w[:self.Position_G.shape[0] *self.Position_G.shape[1]] * (hu_Simpoints * self.first_derivative_covariance_function_divided_by_r(sed_dips_SimPoint))
        
        sigma_0_grad = torch.sum(sigma_0_grad,axis=0)
        

        sed_rest_SimPoint = self.squared_euclidean_distance(self.rest_layer_points,grid_data_plus_ref,Transformation_matrix=Transformation_matrix)
        sed_ref_SimPoint = self.squared_euclidean_distance(self.ref_layer_points,grid_data_plus_ref, Transformation_matrix=Transformation_matrix)

        
        sigma_0_interf =  self.w[self.Position_G.shape[0]*self.Position_G.shape[1]:]*(-self.covariance_function(sed_rest_SimPoint) + self.covariance_function(sed_ref_SimPoint))
        sigma_0_interf = torch.sum(sigma_0_interf,axis = 0)
        
        

        interpolate_result = sigma_0_grad+ sigma_0_interf
        #print("interpolate_result ", interpolate_result.shape)
        scalar_field={}
        scalar_field["scalar_ref_points"] = interpolate_result[-self.number_of_layer:]
        if section_plot == False:
            start = 0
            for keys, values in self.grid_status.items():
                if values is not None:
                    end = start + self.data[keys].shape[0] ####
                    scalar_field[keys] = interpolate_result[start:end]
                    start = end
        else:
            scalar_field["Regular"] = interpolate_result[:-self.number_of_layer]
        
        #print(scalar_field["Regular"].shape)
        
        # scalar_ref_points = interpolate_result[-self.number_of_layer:]
        # interpolate_result_grid =interpolate_result[:-self.number_of_layer]
        
        # labels = torch.ones(interpolate_result_grid.shape[0])
        # i=1
        # label_index=[]
        # for ele in torch.sort(scalar_ref_points)[0]:
        #     label_index.append(i)
        #     mask = interpolate_result_grid >= ele
        #     labels= torch.where(mask, i+1, labels)
        #     i=i+1
            
    
        sorted_ref = torch.cat((interpolate_result.min().unsqueeze(0),torch.sort(scalar_field["scalar_ref_points"])[0], interpolate_result.max().unsqueeze(0)))
    
        #print(sorted_ref)
        
        modified_interpolate_results_final = torch.zeros(interpolate_result.shape)

        for i in range(len(sorted_ref)-2):
            modified_interpolate_results = torch.zeros(interpolate_result.shape)
            
            mask = (interpolate_result >= sorted_ref[i+1]-self.s_1) & (interpolate_result < sorted_ref[i+1]+ self.s_2)
            
            modified_interpolate_results =torch.where(mask, (interpolate_result - sorted_ref[i+1]+self.s_1)/(self.s_1 + self.s_2) , modified_interpolate_results)
            mask =  interpolate_result >  sorted_ref[i+1]+ self.s_2
            modified_interpolate_results =torch.where(mask, 1 , modified_interpolate_results)
            
            modified_interpolate_results_final +=  modified_interpolate_results
            
        modified_interpolate_results_final += 1
        results={}
        results["ref_points"] = modified_interpolate_results_final[-self.number_of_layer:]
        if section_plot == False:
            start = 0
            for keys, values in self.grid_status.items():
                if values is not None:
                    end = start + self.data[keys].shape[0]
                    results[keys] = modified_interpolate_results_final[start:end]
                    start = end
        else:
            results["Regular"] = modified_interpolate_results_final[:-self.number_of_layer]
                
        return scalar_field, results
        
    def Solution(self,Transformation_matrix=None ):
        
        grid_data_final =[]
        
        for keys, values in self.grid_status.items():
            if values is not None:
               grid_data_final.append(self.data[keys]) 
                 
        if len(grid_data_final)>1:
            grid_data_ = torch.cat(grid_data_final, dim=0)
        elif len(grid_data_final)==1:
                grid_data_ = grid_data_final[0]
        else: 
            print("No grid is active")
            grid_data_ = None
            
        
        if Transformation_matrix is None:
            Transformation_matrix = torch.eye(self.op_coord["Positions"].shape[1], dtype =self.dtype)
        
        
        self.scalar_field, self.results = self.Solution_grid(grid_coord=grid_data_, Transformation_matrix=Transformation_matrix)   
        self.solution={}
        self.solution["scalar_field"]= self.scalar_field
        self.solution["result"]=self.results
        return self.solution
    
    def plot_2D(self, data ,sclar_field,  value, plot_scalar_field = True, plot_input_data=True,section=None):
        
        import matplotlib.pyplot as plt
        scatter =plt.scatter(data[:,0], data[:,1], c=value, cmap='viridis', s=100)
                
        axis_label = ["X", "Y", "Z", "T"]
        
        if section is None:
            accepted_index = [0,1]
        else:
            # Get the key to remove (convert to 0-based index)
            remove_index = list(section.keys())[0] - 1

            # Create a new list excluding that index
            accepted_index = [i for i, _ in enumerate(axis_label) if i != remove_index]
            
        if plot_scalar_field:
            X = self.mesh[0].numpy()
            Y = self.mesh[1].numpy()
            Z = sclar_field.reshape(X.shape).numpy()
            plt.contour(X, Y, Z)
        
        # Create a legend
        
        import numpy as np
        legend_labels = np.unique(value.numpy())  # Extract unique labels
        label_map={}
        label_map[1] = "Basement"
        i=1
        for keys, _ in self.sp_coord.items():
            label_map[i+1] = keys
            i = i+1 
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10, label=label_map[label]) for label in legend_labels]
        plt.legend(handles=legend_handles, title='Layers')
        plt.xlabel(axis_label[accepted_index[0]] + " Coordinates")
        plt.ylabel(axis_label[accepted_index[1]] + " Coordinates")
        plt.title('Scatter Plot with Color Labels')
        
        ########################################################################################
        ##### Plot surface points and gradients
        ########################################################################################
        if plot_input_data:
            colour = ['ro', 'bo']
            i=0
            for _, values in self.sp_coord.items():
                plt.plot(values[:,accepted_index[0]], values[:,accepted_index[1]], colour[i])
                i=i+1
            
            for i in range(self.Position_G.shape[0]):
                plt.plot(self.Position_G[i,accepted_index[0]], self.Position_G[i,accepted_index[1]], 'go')
                plt.quiver([self.Position_G[i,accepted_index[0]]],[self.Position_G[i,accepted_index[1]]],self.Value_G[i][accepted_index[0]],self.Value_G[i][accepted_index[1]],color='r')
        plt.savefig("Plot_2D.png")
        plt.close()
        
    def plot_3D(self, data ,  value, plot_scalar_field = True, plot_input_data=True, section=None):
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
        
        fig = plt.figure()
          
        ax = fig.add_subplot(111, projection='3d')

        points = data.numpy()  # shape: (N, 3)
        values = value.numpy()  # shape: (N,)

        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap='viridis', s=0.1, alpha=0.5)
        # Create divider and append colorbar axis
    
        # Optional colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Label")
        import numpy as np
        legend_labels = np.unique(value.numpy())  # Extract unique labels
        label_map={}
        label_map[1] = "Basement"
        i=1
        for keys, _ in self.sp_coord.items():
            label_map[i+1] = keys
            i = i+1 
        
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10, label=label_map[label]) for label in legend_labels]
        plt.legend(handles=legend_handles, title='Layers',loc="upper left")
        
        ################################################################################
        #                                     TODO                                     #
        ################################################################################
        axis_label = ["X", "Y", "Z", "T"]
        
        if section is None:
            accepted_index = [0,1,2]
        else:
            # Get the key to remove (convert to 0-based index)
            remove_index = list(section.keys())[0] - 1

            # Create a new list excluding that index
            accepted_index = [i for i, _ in enumerate(axis_label) if i != remove_index]

            
        ax.set_xlabel(axis_label[accepted_index[0]])
        ax.set_ylabel(axis_label[accepted_index[1]])
        ax.set_zlabel(axis_label[accepted_index[2]], labelpad=-5)
            
        ax.set_title("3D Scatter Plot with Rounded Labels")
        
        
        ########################################################################################
        ##### Plot surface points and gradients
        ########################################################################################
        if plot_input_data:
            
            colour = ['ro', 'bo']
            i=0
            for _, values in self.sp_coord.items():
                ax.plot(values[:,accepted_index[0]], values[:,accepted_index[1]], values[:,accepted_index[2]], colour[i])
                i=i+1
            
            for i in range(self.Position_G.shape[0]):
                ax.plot(self.Position_G[i,accepted_index[0]], self.Position_G[i,accepted_index[1]],self.Position_G[i,accepted_index[2]], 'go')
                ax.quiver([self.Position_G[i,accepted_index[0]]],[self.Position_G[i,accepted_index[1]]],[self.Position_G[i,accepted_index[2]]],self.Value_G[i][accepted_index[0]],self.Value_G[i][accepted_index[1]],self.Value_G[i][accepted_index[2]],color='r')
        plt.savefig("Plot_3D.png")
        plt.close()
        
        
    def plot_data(self, sol, plot_scalar_field = True, plot_input_data=True):
        
        if self.grid_status["Regular"] is not None:
            
            print("Regular grid is activate")
            
            self.dim = self.data["Regular"].shape[1]
            
            if self.dim == 2:
                print("Data has dimension 2 ")
                self.plot_2D(data=self.data["Regular"], sclar_field=sol["scalar_field"]["Regular"], value= torch.round(sol["result"]['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data)
                
                
            if self.dim==3:
                self.plot_3D(data=self.data["Regular"], value= torch.round(sol["result"]['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data)
                
            else:
                print("Since it is difficult to plot more than 3D for visualisation, provide the section along which you want to visualize to make it 2D or 3D plot")
                
    def plot_data_section(self, section, plot_scalar_field = True, plot_input_data=True):
        
        if self.grid_status["Regular"] is not None:
            
            print("Regular grid is activate")
            
            self.dim = self.data["Regular"].shape[1]
            
            if section is not None: 
                plot_dim = self.dim - len(section)
                print("Dimension of the plot would be : ", plot_dim)
                if plot_dim not in [2,3]:
                    print("Plot Dimesnion must be 2 or 3")
                    exit()
                else:
                    self.grid_hyperplane = []

                    # Generate linspace per dimension
                    for i in range(len(self.resolution)):
                        if i+1 in section:
                            self.grid_hyperplane.append(torch.tensor(section[i+1], dtype=self.dtype))
                        else:
                            start = self.extent[2*i]
                            end = self.extent[2*i + 1]
                            self.grid_hyperplane.append(torch.linspace(start, end, self.resolution[i], dtype=self.dtype))
                    # Create meshgrid (cartesian product of all dimensions)
                    mesh = torch.meshgrid(*self.grid_hyperplane, indexing="ij")  # Use 'ij' to keep dimensionality consistent
                    
                    # Flatten and stack each dimension
                    flat = [m.reshape(-1) for m in mesh]
                    full_grid_hyp = torch.stack(flat, dim=1)  # Shape: (num_points, num_dimensions)

                    # Find the solution at this location
                    scalar_field, results = self.Solution_grid(grid_coord=full_grid_hyp, Transformation_matrix=torch.eye(4, dtype=self.dtype),section_plot=True)
                    
                    # Plot
                    columns_to_keep = [i for i in range(full_grid_hyp.shape[1]) if i + 1 not in section]
                    final_grid = full_grid_hyp[:, columns_to_keep]
                    
                    mesh_selected = tuple(mesh[i].squeeze() for i in columns_to_keep)
                    self.mesh = mesh_selected
                    if plot_dim==2:
                        #print(final_grid.shape, scalar_field["Regular"] )
                        
                        
                        self.plot_2D(data=final_grid, sclar_field=scalar_field["Regular"], value= torch.round(results['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data, section=section)
                    else:
                        self.plot_3D(data=final_grid, value= torch.round(results['Regular']), plot_scalar_field=plot_scalar_field, plot_input_data=plot_input_data,section=section) 
                        
    
                
            else:
                print("Provide a valid section")
def main():        
    ###########################################################################################################################      
    #                               2D
    ###########################################################################################################################          
    # gp = Gempy("Gempy_test", 
    #            extent=[-0.5,4.5, -0.5 ,4.5],
    #            resolution=[100, 100]
    #            )

    # interface_data={"Surface 1": torch.tensor([[0,2],[2,4],[4,2],[2,0]]), "Surface 2": torch.tensor([[1,2],[2,3],[3,2],[2,1]])}
    # orientation_data ={"Positions": torch.tensor([[2., 4.0], [2,0], [2,3],[2,1]]), "Values": torch.tensor([[0, 1.0], [0.,-1.0], [0, 1.0], [0.,-1.0]])}
    # gp.interface_data(interface_data)
    # gp.orientation_data(orientation_data)
    # custom_data = torch.tensor([[4,2]], dtype=torch.float32)

    ###########################################################################################################################      
    #                               3D
    ###########################################################################################################################          
    # gp = Gempy("Gempy_test", 
    #            extent=[-0.5,4.5, -0.5 ,4.5, -0.5, 4.5],
    #            resolution=[50, 50, 50]
    #            )

    # interface_data={"Surface 1": torch.tensor([[0,2,1],[2,4,1],[4,2,1],[2,0,1]]), "Surface 2": torch.tensor([[1,2,2],[2,3,2],[3,2,2],[2,1,2]])}
    # orientation_data ={"Positions": torch.tensor([[2., 4.0, 1], [2,0,1], [2,3,2],[2,1,2]]), "Values": torch.tensor([[0, 1.0,0], [0.,-1.0,0], [0, 1.0,0], [0.,-1.0,0]])}
    # gp.interface_data(interface_data)
    # gp.orientation_data(orientation_data)
    # custom_data = torch.tensor([[4,2,3]], dtype=torch.float32)
    ###########################################################################################################################      
    #                               4D
    ###########################################################################################################################          
    gp = Gempy("Gempy_test", 
            extent=[-0.5,4.5, -0.5 ,4.5, -0.5, 4.5, -0.5, 4.5],
            resolution=[50, 50, 50, 50]
            )

    interface_data={"Surface 1": torch.tensor([[0,2,1, 1],[2,4,1,1],[4,2,1,1],[2,0,1,1]]), "Surface 2": torch.tensor([[1,2,2,1],[2,3,2,1],[3,2,2,1],[2,1,2,1]])}
    orientation_data ={"Positions": torch.tensor([[2., 4.0, 1,-1], [2,0,1,1], [2,3,2,2],[2,1,2,3]]), "Values": torch.tensor([[0, 1.0,0,1], [0.,-1.0,0,1], [0, 1.0,0,1], [0.,-1.0,0,1]])}
    gp.interface_data(interface_data)
    gp.orientation_data(orientation_data)
    custom_data = torch.tensor([[4,2,3,1]], dtype=torch.float32)

    gp.activate_custom_grid(custom_grid_data=custom_data)
    #gp.active_grid()
    #gp.deactivate_grid("Regular")
    sol = gp.Solution()
    # print(sol)
    

    #gp.active_grid()
    #gp.activate_grid("Regular")
    gp.active_grid()
    sol = gp.Solution()
    #print(sol)
    # gp.plot_data(sol=sol, plot_scalar_field= True, plot_input_data=True)
    
    for t in [-0.5, 0, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3] :
        gp.plot_data_section(section={3:2, 4:t}, plot_scalar_field = True, plot_input_data=True)
    
    
    # def jacobian_test(*param):
    
    #     gp = Gempy("Gempy_test", 
    #         extent=[-0.5,4.5, -0.5 ,4.5, -0.5, 4.5, -0.5, 4.5],
    #         resolution=[50, 50, 50, 50]
    #         )

    #     interface_data={"Surface 1": torch.tensor([[0,2,1, param[0][0]],[2,4,1,1],[4,2,1,1],[2,0,1,1]]), "Surface 2": torch.tensor([[1,2,2,1],[2,3,2,1],[3,2,2,1],[2,1,2,1]])}
    #     orientation_data ={"Positions": torch.tensor([[2., 4.0, 1,1], [2,0,1,1], [2,3,2,1],[2,1,2,1]]), "Values": torch.tensor([[0, 1.0,0,0], [0.,-1.0,0,0], [0, 1.0,0,0], [0.,-1.0,0,0]])}
    #     gp.interface_data(interface_data)
    #     gp.orientation_data(orientation_data)
    #     #custom_data = torch.tensor([[4,2,3,1]], dtype=torch.float32)
    #     custom_data = torch.randn(1024, 4).to(torch.float32)
    #     gp.activate_custom_grid(custom_grid_data=custom_data)
    #     gp.active_grid()
    #     gp.deactivate_grid("Regular")
    #     sol = gp.Solution()
    #     #print(sol)
        
    #     return sol["result"]["Custom"]
        
    # from torch.autograd.functional import jacobian
    # from datetime import datetime

    # start_time = datetime.now()
    # a=torch.tensor(1.0).reshape(-1,1)
    # print(a.shape)
    # theta_data = tuple([a.clone().requires_grad_(True)])
    # mean = jacobian_test(theta_data)
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time for Gempy: {elapsed_time}")
    # start_time = datetime.now()
    # J = jacobian(jacobian_test, theta_data)
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time for Jacobian: {elapsed_time}")
        
    
if __name__ == "__main__":
    main()