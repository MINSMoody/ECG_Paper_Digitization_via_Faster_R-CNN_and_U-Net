import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from TeamCode.src.ecg_image_generator.TemplateFiles.generate_template import generate_template
from math import ceil 
from PIL import Image
# import csv

standard_values = {'y_grid_size' : 0.5,
                   'x_grid_size' : 0.2,
                   'y_grid_inch' : 5/25.4,
                   'x_grid_inch' : 5/25.4,
                   'grid_line_width' : 0.5,
                   'lead_name_offset' : 0.5,
                   'lead_fontsize' : 11,
                   'x_gap' : 1,
                   'y_gap' : 0.5,
                   'display_factor' : 1,
                   'line_width': 0.75,
                   'row_height' : 8,
                   'dc_offset_length' : 0.2,
                   'lead_length' : 3,
                   'V1_length' : 12,
                   'width' : 11,
                   'height' : 8.5
                   }

standard_major_colors = {'colour1' : (0.4274,0.196,0.1843), #brown
                          'colour2' : (1,0.796,0.866), #pink
                          'colour3' : (0.0,0.0, 0.4), #blue
                          'colour4' : (0,0.3,0.0), #green
                          'colour5' : (1,0,0) #red
    }


standard_minor_colors = {'colour1' : (0.5882,0.4196,0.3960),
                         'colour2' : (0.996,0.9294,0.9725),
                         'colour3' : (0.0,0, 0.7),
                         'colour4' : (0,0.8,0.3),
                         'colour5' : (0.996,0.8745,0.8588)
    }

papersize_values = {'A0' : (33.1,46.8),
                    'A1' : (33.1,23.39),
                    'A2' : (16.54,23.39),
                    'A3' : (11.69,16.54),
                    'A4' : (8.27,11.69),
                    'letter' : (8.5,11)
                    }


def inches_to_dots(value,resolution):
    return (value * resolution)

# Function to add noise to color
def add_noise_to_color(color, noise_level=0.1):
    noisy_color = color + noise_level * np.random.randn(3)
    noisy_color = np.clip(noisy_color, 0, 1)  # Ensure RGB values stay within [0, 1]
    return noisy_color

# Function to adjust pixel intensity
# def adjust_intensity(image_array, x_ticks, y_ticks, intensity_variation=0.2):
    
#     # Modify pixel intensity around grid lines
#     for x in x_ticks:
#         for y in range(image_array.shape[0]):  # for each y pixel
#             variation = np.random.uniform(-intensity_variation, intensity_variation, size=3)
#             image_array[y, int(x)] = np.clip(image_array[y, int(x)] + variation, 0, 255)
    
#     for y in y_ticks:
#         for x in range(image_array.shape[1]):  # for each x pixel
#             variation = np.random.uniform(-intensity_variation, intensity_variation, size=3)
#             image_array[int(y), x] = np.clip(image_array[int(y), x] + variation, 0, 255)
    
#     return Image.fromarray(image_array.astype('uint8'))

def break_line(image_array, x_ticks, y_ticks, line_width=2, probability=0.05):
    
    # Modify pixel intensity around grid lines
    for x in x_ticks:
        for y in range(image_array.shape[0]):  # for each y pixel
            add_y_noise = np.random.choice([True, False], p=[probability, 1-probability])
            if add_y_noise:
                variation = int(np.random.uniform(-line_width, line_width, size=3))
                image_array[y-variation:y+variation, int(x)] = 0
    
    for y in y_ticks:
        for x in range(image_array.shape[1]):  # for each x pixel
            add_x_noise = np.random.choice([True, False], p=[probability, 1-probability])
            if add_x_noise:
                variation = np.random.uniform(-line_width, line_width, size=3)
                image_array[int(y), x-variation:x+variation] = 0
    
    return Image.fromarray(image_array.astype('uint8'))

def adjust_width(image, x_ticks, y_ticks, width_variation=0.2, probability=0.05):
    # 
    image_array = np.array(image)
    
    for x in x_ticks:
        for y in range(image_array.shape[0]):  # for each y pixel
            add_y_noise = np.random.choice([True, False], p=[probability, 1-probability])
            if add_y_noise:
                variation = int(np.random.uniform(-width_variation, width_variation, size=3))
                image_array[y-variation:y+variation, int(x)] = np.clip(image_array[y, int(x)], 0, 255)
    
    for y in y_ticks:
        for x in range(image_array.shape[1]):  # for each x pixe
            add_x_noise = np.random.choice([True, False], p=[probability, 1-probability])
            if add_x_noise:
                variation = np.random.uniform(-width_variation, width_variation, size=3)
                image_array[int(y), x] = np.clip(image_array[int(y), x], 0, 255)
    
    return Image.fromarray(image_array.astype('uint8'))

#Function to plot raw ecg signal
def ecg_plot(
        ecg, 
        configs,
        sample_rate, 
        columns,
        rec_file_name,
        output_dir,
        resolution,
        pad_inches,
        lead_index,
        full_mode,
        store_text_bbox,
        full_header_file,
        units          = '',
        papersize      = '',
        x_gap          = standard_values['x_gap'],
        y_gap          = standard_values['y_gap'],
        display_factor = standard_values['display_factor'],
        line_width     = standard_values['line_width'],
        title          = '',  
        style          = None,
        row_height     = standard_values['row_height'],
        show_lead_name = True,
        show_grid      = False,
        show_dc_pulse  = False,
        y_grid = 0,
        x_grid = 0,
        standard_colours = False,
        bbox = False,
        print_txt=False,
        json_dict=dict(),
        start_index=-1,
        store_configs=0,
        lead_length_in_seconds=10,
        masks=False
        ):
    #Inputs :
    #ecg - Dictionary of ecg signal with lead names as keys
    #sample_rate - Sampling rate of the ecg signal
    #lead_index - Order of lead indices to be plotted
    #columns - Number of columns to be plotted in each row
    #x_gap - gap between paper x axis border and signal plot
    #y_gap - gap between paper y axis border and signal plot
    #line_width - Width of line tracing the ecg
    #title - Title of figure
    #style - Black and white or colour
    #row_height - gap between corresponding ecg rows
    #show_lead_name - Option to show lead names or skip
    #show_dc_pulse - Option to show dc pulse
    #show_grid - Turn grid on or off


    #Initialize some params
    #secs represents how many seconds of ecg are plotted
    #leads represent number of leads in the ecg
    #rows are calculated based on corresponding number of leads and number of columns

    matplotlib.use("Agg")
    line_widths = [line_width * 0.75, line_width, line_width * 1.25]
    line_width_probs = [0.2, 0.5, 0.3]
    chosen_width = np.random.choice(line_widths, p=line_width_probs)

            
    #check if the ecg dict is empty
    if ecg == {}:
        return 

    secs = lead_length_in_seconds

    leads = len(lead_index)

    rows  = int(ceil(leads/columns))

    if(full_mode!='None'):
        rows+=1
        leads+=1
    
    #Grid calibration
    #Each big grid corresponds to 0.2 seconds and 0.5 mV
    #To do: Select grid size in a better way
    y_grid_size = standard_values['y_grid_size']
    x_grid_size = standard_values['x_grid_size']
    grid_line_width = standard_values['grid_line_width']
    lead_name_offset = standard_values['lead_name_offset']
    lead_fontsize = standard_values['lead_fontsize']


    #Set max and min coordinates to mark grid. Offset x_max slightly (i.e by 1 column width)

    if papersize=='':
        width = standard_values['width']
        height = standard_values['height']
    else:
        width = papersize_values[papersize][1]
        height = papersize_values[papersize][0]
    
    y_grid = standard_values['y_grid_inch'] 
    x_grid = standard_values['x_grid_inch']
    y_grid_dots = y_grid*resolution
    x_grid_dots = x_grid*resolution
 
    #row_height = height * y_grid_size/(y_grid*(rows+2))
    row_height = (height * y_grid_size/y_grid)/(rows+2)
    x_max = width * x_grid_size / x_grid
    x_min = 0
    x_gap = np.floor(((x_max - (columns*secs))/2)/0.2)*0.2
    y_min = 0
    y_max = height * y_grid_size/y_grid

    json_dict['width'] = int(width*resolution)
    json_dict['height'] = int(height*resolution)
    #Set figure and subplot sizes
    fig, ax = plt.subplots(figsize=(width, height), dpi=resolution)
    
    
   
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  
        right  = 1,  
        bottom = 0,  
        top    = 1
        )

    fig.suptitle(title)
    
    # Haoliang
    if(masks):
        fig1, ax1 = plt.subplots(figsize=(width, height))
   
        fig1.subplots_adjust(
            hspace = 0, 
            wspace = 0,
            left   = 0,  
            right  = 1,  
            bottom = 0,  
            top    = 1
            )
        fig1.suptitle(title)
        fig1.patch.set_facecolor('black')

        # Set the background color of the axes to black
        ax1.set_facecolor('black')   

    #Mark grid based on whether we want black and white or colour
    
    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    elif(standard_colours > 0):
        random_colour_index = standard_colours
        color_major = standard_major_colors['colour'+str(random_colour_index)]
        color_minor = standard_minor_colors['colour'+str(random_colour_index)]
        grey_random_color = random.uniform(0,0.2)
        color_line  = (grey_random_color,grey_random_color,grey_random_color)
    else:
        major_random_color_sampler_red = random.uniform(0.3,0.8)
        major_random_color_sampler_green = random.uniform(0,0.3)
        major_random_color_sampler_blue = random.uniform(0,0.3)

        minor_offset = random.uniform(0,0.2)
        minor_random_color_sampler_red = major_random_color_sampler_red + minor_offset
        minor_random_color_sampler_green = random.uniform(0,0.5) + minor_offset
        minor_random_color_sampler_blue = random.uniform(0,0.5) + minor_offset

        grey_random_color = random.uniform(0,0.2)
        color_major = (major_random_color_sampler_red,major_random_color_sampler_green,major_random_color_sampler_blue)
        color_minor = (minor_random_color_sampler_red,minor_random_color_sampler_green,minor_random_color_sampler_blue)
        
        color_line  = (grey_random_color,grey_random_color,grey_random_color)

    #Set grid
    #Standard ecg has grid size of 0.5 mV and 0.2 seconds. Set ticks accordingly
    
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    if(masks):
        ax1.grid(False)
        ax1.set_ylim(y_min,y_max)
        ax1.set_xlim(x_min,x_max)
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')    
    #Step size will be number of seconds per sample i.e 1/sampling_rate
    step = (1.0/sample_rate)

    dc_offset = 0
    if(show_dc_pulse):
        dc_offset = sample_rate*standard_values['dc_offset_length']*step
    #Iterate through each lead in lead_index array.
    y_offset = (row_height/2)
    x_offset = 0

    leads_ds = []

    leadNames_12 = configs['leadNames_12']
    tickLength = configs['tickLength']
    tickSize_step = configs['tickSize_step']

    for i in np.arange(len(lead_index)):
        current_lead_ds = dict()

        if len(lead_index) == 12:
            leadName = leadNames_12[i]
        else:
            leadName = lead_index[i]
        #y_offset is computed by shifting by a certain offset based on i, and also by row_height/2 to account for half the waveform below the axis
        if(i%columns==0):

            y_offset += row_height
        
        #x_offset will be distance by which we shift the plot in each iteration
        if(columns>1):
            x_offset = (i%columns)*secs
            
        else:
            x_offset = 0

        #Create dc pulse wave to plot at the beginning of plot. Dc pulse will be 0.2 seconds
        x_range = np.arange(0,sample_rate*standard_values['dc_offset_length']*step + 4*step,step)
        dc_pulse = np.ones(len(x_range))
        dc_pulse = np.concatenate(((0,0),dc_pulse[2:-2],(0,0)))

        #Print lead name at .5 ( or 5 mm distance) from plot
        if(show_lead_name):
            t1 = ax.text(x_offset + x_gap + dc_offset, 
                    y_offset-lead_name_offset - 0.2, 
                    leadName, 
                    fontsize=lead_fontsize)
            
            if (store_text_bbox):
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent()    
                x1 = bb.x0*resolution/fig.dpi      
                y1 = bb.y0*resolution/fig.dpi   
                x2 = bb.x1*resolution/fig.dpi     
                y2 = bb.y1*resolution/fig.dpi    
                box_dict = dict()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_dict[0] = [round(json_dict['height'] - y2, 2), round(x1, 2)]
                box_dict[1] = [round(json_dict['height'] - y2, 2), round(x2, 2)]
                box_dict[2] = [round(json_dict['height'] - y1, 2), round(x2, 2)]
                box_dict[3] = [round(json_dict['height'] - y1, 2), round(x1, 2)]
                current_lead_ds["text_bounding_box"] = box_dict

        current_lead_ds["lead_name"] = leadName

        #If we are plotting the first row-1 plots, we plot the dc pulse prior to adding the waveform
        if(columns == 1 and i in np.arange(0,rows)):
            if(show_dc_pulse):
                #Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(x_range + x_offset + x_gap,
                        dc_pulse+y_offset,
                        linewidth=line_width * 1.5, 
                        color=color_line
                        )
                if (bbox):
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()                                                
                    x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                    x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
                    # don't include dc pulse in the bounding box
                    x1 += len(x_range)
                    
                
        elif(i%columns == 0):
            if(show_dc_pulse):
                #Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(np.arange(0,sample_rate*standard_values['dc_offset_length']*step + 4*step,step) + x_offset + x_gap,
                        dc_pulse+y_offset,
                        linewidth=line_width * 1.5, 
                        color=color_line
                        )
                if (bbox):
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()                                                
                    x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                    x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
                    # don't include dc pulse in the bounding box
                    x1 += len(x_range)

        t1 = ax.plot(np.arange(0,len(ecg[leadName])*step,step) + x_offset + dc_offset + x_gap, 
                ecg[leadName] + y_offset,
                linewidth=line_width, 
                color=color_line
                )
        if(masks):
            t2 = ax1.plot(np.arange(0,len(ecg[leadName])*step,step) + x_offset + dc_offset + x_gap, 
                ecg[leadName] + y_offset,
                linewidth=line_width, 
                color=(1,1,1)
                )
        
        
        x_vals = np.arange(0,len(ecg[leadName])*step,step) + x_offset + dc_offset + x_gap
        y_vals = ecg[leadName] + y_offset

        if (bbox):
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = t1[0].get_window_extent()  
            if show_dc_pulse == False or (columns == 4 and (i != 0 and i != 4 and i != 8)):                                           
                x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
            else:
                y1 = min(y1, bb.y0*resolution/fig.dpi)
                y2 = max(y2, bb.y1*resolution/fig.dpi)
                x2 = bb.x1*resolution/fig.dpi
            box_dict = dict()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            box_dict[0] = [round(json_dict['height'] - y2, 2), round(x1, 2)]
            box_dict[1] = [round(json_dict['height'] - y2, 2), round(x2, 2)]
            box_dict[2] = [round(json_dict['height'] - y1, 2), round(x2, 2)]
            box_dict[3] = [round(json_dict['height'] - y1, 2), round(x1, 2)]
            current_lead_ds["lead_bounding_box"] = box_dict
        
        st = start_index
        if columns == 4 and leadName in configs['format_4_by_3'][1]:
            st = start_index + int(sample_rate*configs['paper_len']/columns)
        elif columns == 4 and leadName in configs['format_4_by_3'][2]:
            st = start_index + int(2*sample_rate*configs['paper_len']/columns)
        elif columns == 4 and leadName in configs['format_4_by_3'][3]:
            st = start_index + int(3*sample_rate*configs['paper_len']/columns)
        current_lead_ds["start_sample"] = st
        current_lead_ds["end_sample"]= st + len(ecg[leadName])
        current_lead_ds["plotted_pixels"] = []
        for j in range(len(x_vals)):
            xi, yi = x_vals[j], y_vals[j]
            xi, yi = ax.transData.transform((xi, yi))
            yi = json_dict['height'] - yi
            current_lead_ds['plotted_pixels'].append([round(yi, 2), round(xi, 2)])

        leads_ds.append(current_lead_ds)

        if columns > 1 and (i+1)%columns != 0:
            sep_x = [len(ecg[leadName])*step + x_offset + dc_offset + x_gap] * round(tickLength*y_grid_dots)
            sep_x = np.array(sep_x)
            sep_y = np.linspace(y_offset - tickLength/2*y_grid_dots*tickSize_step, y_offset + tickSize_step*y_grid_dots*tickLength/2, len(sep_x))
            ax.plot(sep_x, sep_y, linewidth=line_width * 3, color=color_line)

    #Plotting longest lead for 12 seconds
    if(full_mode!='None'):
        current_lead_ds = dict()
        if(show_lead_name):
            t1 = ax.text(x_gap + dc_offset, 
                    row_height/2-lead_name_offset, 
                    full_mode, 
                    fontsize=lead_fontsize)
            
            if (store_text_bbox):
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent(renderer = fig.canvas.renderer)
                x1 = bb.x0*resolution/fig.dpi      
                y1 = bb.y0*resolution/fig.dpi   
                x2 = bb.x1*resolution/fig.dpi     
                y2 = bb.y1*resolution/fig.dpi           
                box_dict = dict()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_dict[0] = [round(json_dict['height'] - y2, 2), round(x1, 2)]
                box_dict[1] = [round(json_dict['height'] - y2, 2), round(x2, 2)]
                box_dict[2] = [round(json_dict['height'] - y1, 2), round(x2, 2)]
                box_dict[3] = [round(json_dict['height'] - y1), round(x1, 2)]
                current_lead_ds["text_bounding_box"] = box_dict                
            current_lead_ds["lead_name"] = full_mode

        if(show_dc_pulse):
            t1 = ax.plot(x_range + x_gap,
                    dc_pulse + row_height/2-lead_name_offset + 0.8,
                    linewidth=line_width * 1.5, 
                    color=color_line
                    )
            
            if (bbox):
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()                                                
                    x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                    x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
                    x1 += len(x_range)
        
        dc_full_lead_offset = 0 
        if(show_dc_pulse):
            dc_full_lead_offset = sample_rate*standard_values['dc_offset_length']*step
        
        t1 = ax.plot(np.arange(0,len(ecg['full'+full_mode])*step,step) + x_gap + dc_full_lead_offset, 
                    ecg['full'+full_mode] + row_height/2-lead_name_offset + 0.8,
                    linewidth=line_width, 
                    color=color_line
                    )
        if(masks):
            t2 = ax1.plot(np.arange(0,len(ecg['full'+full_mode])*step,step) + x_gap + dc_full_lead_offset, 
                    ecg['full'+full_mode] + row_height/2-lead_name_offset + 0.8,
                    linewidth=line_width, 
                    color=(1,1,1)
                    )
        x_vals = np.arange(0,len(ecg['full'+full_mode])*step,step) + x_gap + dc_full_lead_offset
        y_vals = ecg['full'+full_mode] + row_height/2-lead_name_offset + 0.8

        if (bbox):
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = t1[0].get_window_extent()  
            if show_dc_pulse == False:                                           
                x1, y1 = bb.x0*resolution/fig.dpi, bb.y0*resolution/fig.dpi
                x2, y2 = bb.x1*resolution/fig.dpi, bb.y1*resolution/fig.dpi
            else:
                y1 = min(y1, bb.y0*resolution/fig.dpi)
                y2 = max(y2, bb.y1*resolution/fig.dpi)
                x2 = bb.x1*resolution/fig.dpi
                x1 += dc_full_lead_offset

            box_dict = dict()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            box_dict[0] = [round(json_dict['height'] - y2, 2), round(x1, 2)]
            box_dict[1] = [round(json_dict['height'] - y2), round(x2, 2)]
            box_dict[2] = [round(json_dict['height'] - y1, 2), round(x2, 2)]
            box_dict[3] = [round(json_dict['height'] - y1, 2), round(x1, 2)]
            current_lead_ds["lead_bounding_box"] = box_dict
        current_lead_ds["start_sample"] = start_index
        current_lead_ds["end_sample"] = start_index + len(ecg['full'+full_mode])
        current_lead_ds['plotted_pixels'] = []
        for i in range(len(x_vals)):
            xi, yi = x_vals[i], y_vals[i]
            xi, yi = ax.transData.transform((xi, yi))
            yi = json_dict['height'] - yi
            current_lead_ds['plotted_pixels'].append([round(yi, 2), round(xi, 2)])
        leads_ds.append(current_lead_ds)



    head, tail = os.path.split(rec_file_name)
    rec_file_name = os.path.join(output_dir, tail)

    #printed template file
    if print_txt:
        x_offset = 0.05
        y_offset = int(y_max)
        printed_text, attributes, flag = generate_template(full_header_file)

        if flag:
            for l in range(0, len(printed_text), 1):
        
                for j in printed_text[l]:
                    curr_l = ''
                    if j in attributes.keys():
                        curr_l += str(attributes[j])
                    ax.text(x_offset, y_offset, curr_l, fontsize=lead_fontsize)
                    x_offset += 3

                y_offset -= 0.5
                x_offset = 0.05
        else:
            for line in printed_text:
                ax.text(x_offset, y_offset, line, fontsize=lead_fontsize)
                y_offset -= 0.5

    #change x and y res
    ax.text(2, 0.5, '25mm/s', fontsize=lead_fontsize)
    ax.text(4, 0.5, '10mm/mV', fontsize=lead_fontsize)
    
    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,x_grid_size))    
        ax.set_yticks(np.arange(y_min,y_max,y_grid_size))
        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        #set grid line style
        ax.grid(which='major', linestyle='-', linewidth=grid_line_width, color=color_major)
        
        ax.grid(which='minor', linestyle='-', linewidth=grid_line_width, color=color_minor)
        
        # Retrieve all gridlines
        lines = ax.get_xgridlines() + ax.get_ygridlines()

        # Precompute the probabilities for line width and disappearance
        line_widths = [grid_line_width * 0.75, grid_line_width, grid_line_width * 1.5]
        line_width_probs = [0.2, 0.5, 0.3]
        solid_line_prob = 0.3

        # Iterate over the gridlines
        for line in lines:
            # Set the line width based on precomputed probabilities
            chosen_width = np.random.choice(line_widths, p=line_width_probs)
            
            # Decide if the line should disappear
            if np.random.random() < 0.15:
                line.set_linewidth(0)  # Disappear the line
            else:
                line.set_linewidth(chosen_width)
            
            # Decide if the line should be solid
            if np.random.random() < solid_line_prob:
                line.set_color((0, 0, 0))  # Set the line to black

        intersection_strength = 1.75  # Amplify the intersection points
        if np.random.random() < 0.2:
             # Emphasize the intersections
            for i in np.arange(x_min, x_max, x_grid_size):
                for j in np.arange(y_min, y_max, y_grid_size):
                    ax.plot(i, j, 'o', markersize=grid_line_width * intersection_strength, color=color_major)

        # break_line(ax, np.arange(x_min,x_max,x_grid_size), np.arange(y_min,y_max,y_grid_size), line_width=2, probability=0.05)
        # adjust_width(ax, np.arange(x_min,x_max,x_grid_size), np.arange(y_min,y_max,y_grid_size), width_variation=0.2, probability=0.05)
        
        
        
        if store_configs == 2:
            json_dict['grid_line_color_major'] = [round(x*255., 2) for x in color_major]
            json_dict['grid_line_color_minor'] = [round(x*255., 2) for x in color_minor]
            json_dict['ecg_plot_color'] = [round(x*255., 2) for x in color_line]
    else:
        ax.grid(False)
    
    fig.savefig(os.path.join(output_dir,tail +'.png'),dpi=resolution)
    plt.close(fig)
    fig.clf()
    ax.cla()

    if(masks):
        file_path = os.path.join(output_dir, 'masks', tail + '.png')  # Final file path

        # Ensure the masks directory exists
        masks_dir = os.path.join(output_dir, 'masks')
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

        # Save the current figure to a temporary file
        
        fig1.savefig(file_path, dpi=resolution)
        plt.close(fig1)
        fig1.clf()
        ax1.cla()

    if pad_inches!=0:
        ecg_image = Image.open(os.path.join(output_dir,tail +'.png'))
        mask_image = Image.open(os.path.join(output_dir,'masks',tail + '.png'))
        
        right = pad_inches * resolution
        left = pad_inches * resolution
        top = pad_inches * resolution
        bottom = pad_inches * resolution
        width, height = ecg_image.size
        
        pad_mode = np.random.choice([1, 2, 3, 4], p=[0.5, 0.15, 0.15, 0.2])
        if pad_mode == 1:
            new_width = width + right + left
            new_height = height + top + bottom
            result_image = Image.new(ecg_image.mode, (new_width, new_height), (255, 255, 255))
            result_image.paste(ecg_image, (left, top))
            result_mask = Image.new(mask_image.mode, (new_width, new_height), (0, 0, 0))
            result_mask.paste(mask_image, (left, top))
            for lead in leads_ds:
                lead_bbox = lead['lead_bounding_box']
                for key, value in lead_bbox.items():
                    lead_bbox[key] = [round(int(value[0]) + top), round(int(value[1]) + left)]
        elif pad_mode == 2:
            new_width = width + right + left
            new_height = height
            result_image = Image.new(ecg_image.mode, (new_width, new_height), (255, 255, 255))
            result_image.paste(ecg_image, (left, 0))
            result_mask = Image.new(mask_image.mode, (new_width, new_height), (0, 0, 0))
            result_mask.paste(mask_image, (left, 0))
            for lead in leads_ds:
                lead_bbox = lead['lead_bounding_box']
                for key, value in lead_bbox.items():
                    lead_bbox[key] = [round(int(value[0])), round(int(value[1]) + left)]
    
        elif pad_mode == 3:
            new_width = width
            new_height = height + top + bottom
            result_image = Image.new(ecg_image.mode, (new_width, new_height), (255, 255, 255))
            result_image.paste(ecg_image, (0, top))
            result_mask = Image.new(mask_image.mode, (new_width, new_height), (0, 0, 0))
            result_mask.paste(mask_image, (0, top))
            for lead in leads_ds:
                lead_bbox = lead['lead_bounding_box']
                for key, value in lead_bbox.items():
                    lead_bbox[key] = [round(int(value[0]) + top), round(int(value[1]))]
        else:
            new_width = width + right
            new_height = height + top 
            result_image = Image.new(ecg_image.mode, (new_width, new_height), (255, 255, 255))
            result_image.paste(ecg_image, (int(left//2), int(top//2)))
            result_mask = Image.new(mask_image.mode, (new_width, new_height), (0, 0, 0))
            result_mask.paste(mask_image, (int(left//2), int(top//2)))
            for lead in leads_ds:
                lead_bbox = lead['lead_bounding_box']
                for key, value in lead_bbox.items():
                    lead_bbox[key] = [round(int(value[0]) + top//2), round(int(value[1]) + left//2)]
            
        
        
        result_image.save(os.path.join(output_dir,tail +'.png'))
        
        
        
        result_mask.save(os.path.join(output_dir,'masks',tail + '.png'))

        plt.close(fig)
        fig.clf()
        ax.cla()
    
    

    json_dict["leads"] = leads_ds

    return x_grid_dots,y_grid_dots
       