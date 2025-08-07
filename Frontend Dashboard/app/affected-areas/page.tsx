"use client"

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Search, Filter, Info, Globe, MapPin, Users, Home, Clock } from 'lucide-react'
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export default function AffectedAreasPage() {
  const initialAffectedAreas = [
    {
      id: "L001",
      name: "Riverbend Community",
      status: "Critical Flood Warning",
      population: "5,200",
      infrastructure: "High Risk",
      lastUpdate: "5 mins ago",
      coordinates: "34.0522° N, 118.2437° W",
    },
    {
      id: "L002",
      name: "Coastal Area A",
      status: "Flood Advisory",
      population: "1,800",
      infrastructure: "Medium Risk",
      lastUpdate: "30 mins ago",
      coordinates: "33.7000° N, 118.0000° W",
    },
    {
      id: "L003",
      name: "Lowlands Sector 7",
      status: "Flood Watch",
      population: "3,100",
      infrastructure: "Low Risk",
      lastUpdate: "1 hour ago",
      coordinates: "34.1000° N, 118.3000° W",
    },
    {
      id: "L004",
      name: "Hillside Village",
      status: "Evacuation Order",
      population: "950",
      infrastructure: "Critical Risk",
      lastUpdate: "Yesterday",
      coordinates: "34.2000° N, 118.4000° W",
    },
    {
      id: "L005",
      name: "Central Plains Farms",
      status: "Resolved",
      population: "200",
      infrastructure: "Low Risk",
      lastUpdate: "2 days ago",
      coordinates: "34.3000° N, 118.5000° W",
    },
  ];

  const [affectedAreas, setAffectedAreas] = useState(initialAffectedAreas);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterRisk, setFilterRisk] = useState('all');

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "Critical Flood Warning":
      case "Evacuation Order":
        return <Badge variant="destructive">{status}</Badge>
      case "Flood Advisory":
        return <Badge variant="default">{status}</Badge>
      case "Flood Watch":
        return <Badge variant="outline">{status}</Badge>
      case "Resolved":
        return <Badge variant="secondary">{status}</Badge>
      default:
        return <Badge>{status}</Badge>
    }
  }

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    console.log("Searching for:", e.target.value);
  };

  const handleStatusFilterChange = (value: string) => {
    setFilterStatus(value);
    console.log("Filtering by status:", value);
  };

  const handleRiskFilterChange = (value: string) => {
    setFilterRisk(value);
    console.log("Filtering by risk:", value);
  };

  const handleViewDetails = (id: string) => {
    alert(`Viewing details for affected area: ${id}`);
    // In a real app, this would open a modal with more info or navigate to a dedicated page
  };

  const filteredAreas = affectedAreas.filter(area => {
    const matchesSearch = area.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          area.status.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || area.status.toLowerCase().includes(filterStatus.toLowerCase());
    const matchesRisk = filterRisk === 'all' || area.infrastructure.toLowerCase().includes(filterRisk.toLowerCase());
    return matchesSearch && matchesStatus && matchesRisk;
  });

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Affected Areas Map</CardTitle>
          <CardDescription>Interactive map showing real-time flood zones, community statuses, and risk levels.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] w-full bg-gray-100 flex items-center justify-center rounded-lg text-muted-foreground border border-dashed">
            <Globe className="h-12 w-12 mr-2" /> Advanced Interactive Map (e.g., with GeoJSON overlays)
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Affected Locations List</CardTitle>
          <CardDescription>Detailed information for each affected community with filtering options.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-4">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search locations by name or status..."
                className="w-full rounded-lg bg-background pl-8"
                value={searchTerm}
                onChange={handleSearchChange}
              />
            </div>
            <Select value={filterStatus} onValueChange={handleStatusFilterChange}>
              <SelectTrigger className="w-[180px] bg-transparent">
                <Filter className="h-3.5 w-3.5 mr-2" />
                <SelectValue placeholder="Filter by Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active Warnings</SelectItem>
                <SelectItem value="advisory">Advisories</SelectItem>
                <SelectItem value="evacuation">Evacuation Orders</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterRisk} onValueChange={handleRiskFilterChange}>
              <SelectTrigger className="w-[180px] bg-transparent">
                <MapPin className="h-3.5 w-3.5 mr-2" />
                <SelectValue placeholder="Filter by Risk" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Risks</SelectItem>
                <SelectItem value="critical">Critical Risk</SelectItem>
                <SelectItem value="high">High Risk</SelectItem>
                <SelectItem value="medium">Medium Risk</SelectItem>
                <SelectItem value="low">Low Risk</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Location</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Population</TableHead>
                <TableHead>Infrastructure Risk</TableHead>
                <TableHead>Last Update</TableHead>
                <TableHead className="text-right">Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAreas.map((area) => (
                <TableRow key={area.id}>
                  <TableCell className="font-medium flex items-center gap-2">
                    <Home className="h-4 w-4 text-muted-foreground" /> {area.name}
                  </TableCell>
                  <TableCell>{getStatusBadge(area.status)}</TableCell>
                  <TableCell className="flex items-center gap-1">
                    <Users className="h-4 w-4 text-muted-foreground" /> {area.population}
                  </TableCell>
                  <TableCell>{area.infrastructure}</TableCell>
                  <TableCell className="flex items-center gap-1">
                    <Clock className="h-3 w-3 text-muted-foreground" /> {area.lastUpdate}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="icon" onClick={() => handleViewDetails(area.id)}>
                      <Info className="h-4 w-4" />
                      <span className="sr-only">View Details</span>
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
